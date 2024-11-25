#!/usr/bin/python3
#coding=utf-8

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lscloss import *

import  torchvision.utils as vutils
import numpy as np
import random

from models.model import get_sod_model
from models.loss import *
from utils.utils import *
from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *
from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *
from tools.ai.augment_utils import *
from tools.ai.randaugment import *

from models.PAR import PAR

import infer
import test_tool
import unss

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def IOU(pred, target):
    inter = target * pred
    union = target + pred - target * pred
    iou_loss = 1 - torch.sum(inter, dim=(1, 2, 3)) / (torch.sum(union, dim=(1, 2, 3)) + 1e-7)
    return iou_loss.mean()

def IOU_Loss(preds, target):
    loss = 0
    
    target = target.gt(0.5).float()
    preds = nn.functional.interpolate(preds, size=target.size()[-2:], mode='bilinear')
    loss += IOU(torch.sigmoid(preds), target)
        
    return loss

loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_lsc_radius = 5
def train(Dataset):
    TAG = "moco_v2_sod_seg"
    BATCH_SIZE = 16
    EPOCHS = 10
    alpha = 0.25

    log_dir = create_directory('./experiments/logs/')
    log_path = log_dir + '{}.txt'.format(TAG)
    model_dir = create_directory('./experiments/models/')
    model_path = model_dir + '{}.pth'.format(TAG)
    cam_path = './experiments/images/{}'.format(TAG)
    create_directory(cam_path)
    create_directory(cam_path + '/train')
    create_directory(cam_path + '/test')
    create_directory(cam_path + '/train/colormaps')
    create_directory(cam_path + '/test/colormaps')

    train_timer = Timer()

    cfg    = Dataset.Config(datapath='./dataset/DUTS-TR/', savepath='./experiments/models/', mode='train', batch=BATCH_SIZE, lr=0.0025, momen=0.9, decay=5e-4, epoch=EPOCHS) # batch=28 # lr = 0.03 -》 0.005  0.003
    data   = Dataset.UData(cfg)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, pin_memory=True, num_workers=4)  

    net = get_sod_model()
    # param_groups = net.get_parameter_groups()

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        print('cuda visible device exception')
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    print(use_gpu)
    if the_number_of_gpu > 1:
        print('preparing data parallel')
        net = nn.DataParallel(net)
    net.train()
    net.cuda()

    par = PAR(num_iter=10, dilations=[1,2,4,8,12,24])
    par.cuda()

    criterion = [SimMaxLoss(metric='cos', alpha=alpha).cuda(), SimMinLoss(metric='cos').cuda(),
                 SimMaxLoss(metric='cos', alpha=alpha).cuda()]

    config = {
        'optim': 'SGD', # 'Adam'
        'lr': 0.005,  # '1e-5'
        'epoch': 15,
        'step_size': [15],
        'gamma': 0.1,
        'clip_gradient': 0,
    }
    module_lr = [
        {'params': net.encoder.parameters(), 'lr': config['lr'] / 10, 'weight_decay': 0.00005},
        {'params': net.ac_head.parameters(), 'lr': config['lr'] / 10, 'weight_decay': 0.00005},
        {'params': net.decoder.parameters(), 'lr': config['lr'], 'weight_decay': 0.0005, 'momentum': 0.9, 'nesterov': True},
    ]
    optimizer = torch.optim.SGD(params=module_lr)

    train_meter = Average_Meter(['loss', 'positive_loss', 'negative_loss', 'bce_loss', 'lsc_loss', 'reg_loss', 'iou_loss'])
    data_dic = {
        'train': [],
        'validation': []
    }
    log_func = lambda string='': log_print(string, log_path)
    flag = False

    CE = torch.nn.BCELoss().cuda()
    loss_lsc = LocalSaliencyCoherence().cuda()

    tmp_path = './moco_see'
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)

    torch.cuda.empty_cache()

    for epoch in range(EPOCHS):

        for i, image in enumerate(loader):
            image = image.type(torch.FloatTensor).cuda()

            optimizer.zero_grad()
            fg_feats, bg_feats, ccam, out_final = net(image) 

            # check and make valid ccam
            if epoch == 0 and i == (len(loader) - 1):
                flag = check_positive(ccam)
                print(f"Is Negative: {flag}")
            if flag:
                ccam = 1 - ccam

            # ccam loss
            loss1 = criterion[0](fg_feats)
            loss2 = criterion[1](bg_feats, fg_feats)
            loss3 = criterion[2](bg_feats)
            ccam_loss = loss1 + loss2 + loss3

            # refine cam
            gt_label, bg_label = infer.light_cam(net, image, flag)
            refined_gt_ori = infer.refine_cam(par, image, gt_label)
            refined_gt = unss.make_unss(refined_gt_ori, rat=2.5)
            # refined_bg = infer.refine_cam(par, image, bg_label)
            gt = gt_label
            mask = gt + bg_label
            mask[mask == 2.] = 1.

            assert 0. <= torch.max(gt) <= 1.
            assert 0. <= torch.max(mask) <= 1.

            # bce loss
            out_final_prob = torch.sigmoid(out_final)
            img_size = image.size(2) * image.size(3) * image.size(0)
            ratio = img_size / torch.sum(mask)
            sal_loss2 = ratio * CE(out_final_prob * mask, gt * mask)

            # reg loss
            # reg_loss = get_energy_loss(img=image, logit=torch.sigmoid(out_final), loss_layer=loss_layer)
            reg_loss = torch.tensor([0.]).cuda()

            # iou loss
            iou_loss = 0.1 * IOU_Loss(out_final, refined_gt)

            # lsc loss
            image_ = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=True)
            sample = {'rgb': image_}
            out_final_prob = F.interpolate(out_final_prob, scale_factor=0.25, mode='bilinear', align_corners=True)
            loss2_lsc = loss_lsc(out_final_prob, loss_lsc_kernels_desc_defaults, 
                                 loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']

            # combine multiple loss
            seg_loss = sal_loss2 + loss2_lsc + iou_loss
            if epoch < 1:
                loss = ccam_loss + 0. * seg_loss
            else: 
                loss = 0.1 * ccam_loss + seg_loss

            loss.backward()
            # clip_gradient(optimizer, config['lr'])
            optimizer.step()

            train_meter.add({
                'loss': loss.item(),
                'positive_loss': loss1.item() + loss3.item(),
                'negative_loss': loss2.item(),
                'bce_loss': sal_loss2.item(),
                'lsc_loss': loss2_lsc.item(),
                'reg_loss': reg_loss.item(),
                'iou_loss': iou_loss.item(),
            })

            if i % 20 == 0:  
                visualize_heatmap(TAG, image.clone().detach(), ccam, 0, i)
                loss, positive_loss, negative_loss, bce_loss, lsc_loss, reg_loss, iou_loss = train_meter.get(clear=True)
                learning_rate = float(get_learning_rate_from_optimizer(optimizer))

                data = {
                    'epoch': epoch,
                    'max_epoch': EPOCHS,
                    'iteration': i + 1,
                    'learning_rate': learning_rate,
                    'loss': loss,
                    'positive_loss': positive_loss,
                    'negative_loss': negative_loss,
                    'bce_loss': bce_loss,
                    'lsc_loss': lsc_loss,
                    'reg_loss': reg_loss,
                    'iou_loss': iou_loss,
                    'time': train_timer.tok(clear=True),
                }
                data_dic['train'].append(data)

                log_func('[i]\t'
                         'Epoch[{epoch:,}/{max_epoch:,}],\t'
                         'iteration={iteration:,}, \t'
                         'learning_rate={learning_rate:.8f}, \t'
                         'loss={loss:.4f}, \t'
                         'positive_loss={positive_loss:.4f}, \t'
                         'negative_loss={negative_loss:.4f}, \t'
                         'bce_loss={bce_loss:.4f}, \t'
                         'lsc_loss={lsc_loss:.4f}, \t'
                         'reg_loss={reg_loss:.4f}, \t'
                         'iou_loss={iou_loss:.4f}, \t'
                         'time={time:.0f}sec'.format(**data)
                         )

                vutils.save_image(torch.sigmoid(ccam[0,:,:,:].data), tmp_path + '/iter%d-ccam.jpg' % i, normalize=True, padding=0)
                vutils.save_image(torch.sigmoid(out_final[0,:,:,:].data), tmp_path + '/iter%d-sal-final.jpg' % i, normalize=True, padding=0)
                vutils.save_image(image[0,:,:,:].data, tmp_path + '/iter%d-sal-image.jpg' % i, padding=0)
                vutils.save_image(bg_label[0,:,:,:].data, tmp_path + '/iter%d-cam-bg.jpg' % i, padding=0)
                vutils.save_image(gt_label[0,:,:,:].data, tmp_path + '/iter%d-cam-gt.jpg' % i, padding=0)
                vutils.save_image(refined_gt[0,:,:,:].data, tmp_path + '/iter%d-refined-gt.jpg' % i, padding=0)

        # do for echo epoch
        torch.save({'state_dict': net.module.state_dict() if (the_number_of_gpu > 1) else net.state_dict(),
                'flag': flag}, cfg.savepath + f'{TAG}-{epoch}.pth')

if __name__=='__main__':
    set_seed(7)
    train(dataset)

