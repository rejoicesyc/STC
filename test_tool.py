#!/usr/bin/python3
#coding=utf-8

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import copy

import cv2
import numpy as np
from tqdm import tqdm
import time

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

from Saliency_Evaluation_numpy.saliency_toolbox import calculate_measures

from models.model import get_sod_model
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


all_dataset = [ './dataset/GT/DUTS_test', './dataset/GT/DUT_O', 
                './dataset/GT/ECSSD', './dataset/GT/HKU_IS', 
                './dataset/GT/PASCAL_S', './dataset/GT/DUTS-TR']

duts_test = ['./dataset/GT/DUTS-TR']


def test(epoch, only_duts_test=True, for_stage2=False):
    dataset = dataset
    assert epoch >= 0

    localtime = time.asctime( time.localtime(time.time()) )

    TAG = "moco_v2_sod_seg"
    experiment_name = TAG
    experiment_name += '@val'
    
    pred_dir = create_directory(f'./experiments/predictions/{experiment_name}/')
    if for_stage2:
        model_path = './out_2nd/' + f'{TAG}-{epoch}.pth'
    else:
        model_path = './experiments/models/' + f'{TAG}-{epoch}.pth'

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    normalize_fn = Normalize(imagenet_mean, imagenet_std)

    scales = [float(scale) for scale in '0.5,1.0,1.5,2.0'.split(',')]

    model = get_sod_model()
    model = model.cuda()
    model.eval()

    ckpt = torch.load(model_path)
    flag = ckpt['flag']
    model.load_state_dict(ckpt['state_dict'])

    model.eval()

    pp = 'DUTS-TR'
    cam_path = create_directory(f'./vis_cam/{experiment_name}/{pp}')
    print(cam_path)

    if only_duts_test:
        record = f'moco_v2_duts_test'
    else: 
        record = f'moco_v2_all_test'

    if for_stage2:
        record = '2nd_' + record
        test_save_path = 'eval'
    else: 
        test_save_path = 'moco_res_out'
    logfile = record + '.txt' # 每测试完一个数据集记录一次

    with open(logfile, 'a') as f:
        f.write("\n------------cut off line--------------\n")
        f.write(str(localtime) + '\n')
        f.write(f'start testing epoch {epoch}\n')

    def test_single(Dataset, Path):
        print(Path)
        
        ## dataset
        cfg    = Dataset.Config(datapath=Path, mode='test') 
        dataset   = Dataset.UData(cfg)
        loader = DataLoader(dataset, batch_size=12, shuffle=False, num_workers=8)

        with torch.no_grad():
            for image, (H, W), name in tqdm(loader):
                image = image.cuda().float()
                _, _, _, out = model(image)

                out = torch.sigmoid(out).cpu().numpy() * 255
                for i in range(out.shape[0]):
                    pred = cv2.resize(out[i, 0], dsize=(int(W[i]),int(H[i])), interpolation=cv2.INTER_LINEAR)
                    head = f'./{test_save_path}/maps/' + cfg.datapath.split('/')[-1]   
                    if not os.path.exists(head):
                        os.makedirs(head)
                    cv2.imwrite(head + '/' + name[i] + '.png', np.round(pred))
        
        if for_stage2:
            method = 'detector'
        else: 
            method = 'usod'
        res = {}
        gt_dir = Path + '/mask'
        datasetname = Path.split('/')[-1]
        sm_dir = f'./{test_save_path}/maps/' + datasetname  # 'SM/'
        if not os.path.exists(sm_dir):
            res[datasetname] = {'Max-F':0, 'Mean-F':0, 'S-measure':0, 'MAE':0, 'Adp-E-measure':0}
            raise ValueError('sm_dir not exist.')

        print('Evaluate ' + method + ' ' + datasetname + '------')

        res[datasetname]=calculate_measures(gt_dir, sm_dir, ['MAE', 'S-measure', 'Max-F', 'Mean-F', 'Adp-E-measure'], 
                                save=False)

        with open(logfile, 'a') as f:  # 'a' 打开文件接着写
            f.write('{} {} get {:.3f} mae, {:.3f} max-f, '
                    '{:.3f} s-measure, {:.3f} e-measure, {:.3f} mean-f \n'.format(
                datasetname, method, res[datasetname]['MAE'], res[datasetname]['Max-F'],
                res[datasetname]['S-measure'], res[datasetname]['Adp-E-measure'], 
                res[datasetname]['Mean-F']))

        return res[datasetname]['MAE'], res[datasetname]['Max-F'], res[datasetname]['S-measure'], \
               res[datasetname]['Adp-E-measure'], res[datasetname]['Mean-F']

    test_list = duts_test if only_duts_test else all_dataset
    for path in test_list:
        ret = test_single(dataset, path)

    return list(ret)
        
if __name__ == '__main__':
    test(7, only_duts_test=False)
