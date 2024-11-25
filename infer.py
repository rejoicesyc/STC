#!/usr/bin/python3
#coding=utf-8

import os

import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import copy

import cv2
import numpy as np
import test_tool

import torch
import torch.nn as nn
import torch.nn.functional as F
import dataset
from torch.utils.data import DataLoader
from utils.utils import *
from models.model import get_model
from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *
from tools.ai.augment_utils import *
from tools.ai.randaugment import *

import cmapy

def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def get_strided_size(orig_size, stride):
    return ((orig_size[0]-1)//stride+1, (orig_size[1]-1)//stride+1)

def check_positive(am):
    assert am.shape[0] == am.shape[1]
    n = am.shape[0]
    edge_mean = (am[0, :].mean() + am[n - 1, :].mean() + am[:, 0].mean() + am[:, n - 1].mean()) / 4
    return edge_mean > 0.5

# scales = [float(scale) for scale in '0.5,1.0,1.5,2.0'.split(',')]

def infer_ccam(model, ori_image, flag, hith=0.5, loth=0.2):
    b, c, h, w = ori_image.shape

    with torch.no_grad():
        def get_cam(image, scale):
            image = F.interpolate(image, size=(int(h * scale), int(w * scale)), 
                                  mode='bilinear', align_corners=True)
        
            image = torch.cat([image, image.flip(-1)], dim=0)
            cams = model(image, cam_only=True)

            if flag:
                cams = 1 - cams

            cams = F.relu(cams)
            cams = cams[:b, ...] + cams[b:, ...].flip(-1)

            return cams

        cams_list = [get_cam(ori_image, scale) for scale in scales]
    
        strided_up_size = get_strided_up_size((h, w), 16)
        hr_cams_list = [
            resize_for_tensors(cams, strided_up_size) for cams in cams_list
        ]
        cam = torch.sum(torch.stack(hr_cams_list, dim=0), dim=0)
        
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5
        cam = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=True)

    fg = torch.zeros_like(cam)
    fg[cam >= hith] = 1.
    bg = torch.zeros_like(cam)
    bg[cam <= loth] = 1.

    return fg, bg

scales = [1, 0.5, 1.5]
# hith, loth = 0.55, 0.15
hith = float(os.environ['HITH'])
loth = float(os.environ['LOTH'])
assert hith > 0. and loth > 0.
high, low = hith, loth

def light_cam(model, inputs, flag,):
    cam_list = []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        _cam = model(inputs_cat, cam_only=True)

        if flag:
            _cam = 1 - _cam

        _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
        
        cam_list = [F.relu(_cam)]

        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam = model(inputs_cat, cam_only=True)

                if flag:
                    _cam = 1 - _cam

                _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))

                cam_list.append(F.relu(_cam))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

        fg = torch.zeros_like(cam)
        fg[cam >= hith] = 1.
        bg = torch.zeros_like(cam)
        bg[cam <= loth] = 1.

    return fg, bg

def _refine_cams(ref_mod, images, cams, valid_key, orig_size):

    refined_cams = ref_mod(images, cams)
    refined_cams = F.interpolate(refined_cams, size=orig_size, 
                                 mode="bilinear", align_corners=False)
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label

def refine_cam(ref_mod, images, cams, down_scale=2):
    b, _, h, w = images.shape
    _images = F.interpolate(images, size=[h//down_scale, w//down_scale], 
                            mode="bilinear", align_corners=False)

    bkg_h = torch.ones(size=(b, 1, h, w)) * high
    bkg_h = bkg_h.to(cams.device)
    bkg_l = torch.ones(size=(b, 1, h, w)) * low
    bkg_l = bkg_l.to(cams.device)

    refined_label = torch.ones(size=(b, h, w))
    refined_label = refined_label.to(cams.device)
    refined_label_h = refined_label.clone()
    refined_label_l = refined_label.clone()

    cams_with_bkg_h = torch.cat((bkg_h, cams), dim=1)
    cams_with_bkg_h = F.interpolate(cams_with_bkg_h, size=[h//down_scale, w//down_scale],
                                     mode="bilinear", align_corners=False)
    cams_with_bkg_l = torch.cat((bkg_l, cams), dim=1)
    cams_with_bkg_l = F.interpolate(cams_with_bkg_l, size=[h//down_scale, w//down_scale], 
                                     mode="bilinear", align_corners=False)

    valid_key = torch.tensor([0, 1])

    for idx in range(b):
        valid_cams_h = cams_with_bkg_h[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)
        valid_cams_l = cams_with_bkg_l[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)

        refined_label_h[idx, ...] = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], 
                                        cams=valid_cams_h, valid_key=valid_key, orig_size=(h, w))[0, ...]
        refined_label_l[idx, ...] = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], 
                                        cams=valid_cams_l,  valid_key=valid_key, orig_size=(h, w))[0, ...]

    refined_label = refined_label_h.clone()
    refined_label[refined_label_h == 0] = 1.
    refined_label[(refined_label_h + refined_label_l) == 0] = 0

    return refined_label.unsqueeze(1)
