import os
import random

from einops import rearrange, repeat, reduce
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import traceback
from tqdm import tqdm
import nibabel as nib


def contains(text, key):
    if isinstance(key, str):
        return key in text
    elif isinstance(key, list):
        for k in key:
            if k in text:
                return True
        return False       
        
class Splited_Images_and_Labels_3D(Dataset):
    def __init__(self, jsonl_file, label_json, batch_size=2, patch_size=[288, 288, 96]):
        """
        针对单一数据集
        拆分depth和query实现并行
        """
        # load data info
        self.jsonl_file = jsonl_file
        with open(self.jsonl_file, 'r') as f:
            lines = f.readlines()
        self.lines = [json.loads(line) for line in lines]
        
        for sample in lines:
            # if train on dbcloud, check the local copy availablity
            path = sample['renorm_image']
            if '/remote-home/share/SAM' in path:
                local_img_path_202 = path.replace('/remote-home/share/SAM', '/remote-home/share/data202/172.16.11.202/SAM')
                if os.path.exists(local_img_path_202):
                    sample['renorm_image'] = local_img_path_202
                else:
                    local_img_path_118 = path.replace('/remote-home/share/SAM', '/remote-home/share/data118/SAM')
                    if os.path.exists(local_img_path_118):
                        sample['renorm_image'] = local_img_path_118
                        
            path = sample['renorm_segmentation_dir']
            if '/remote-home/share/SAM' in path:
                local_img_path_202 = path.replace('/remote-home/share/SAM', '/remote-home/share/data202/172.16.11.202/SAM')
                if os.path.exists(local_img_path_202):
                    sample['renorm_segmentation_dir'] = local_img_path_202
                else:
                    local_img_path_118 = path.replace('/remote-home/share/SAM', '/remote-home/share/data118/SAM')
                    if os.path.exists(local_img_path_118):
                        sample['renorm_segmentation_dir'] = local_img_path_118
        
        with open(label_json, 'r') as f:
            dict = json.load(f)
        c1 = 0
        self.datset_c1c2 = {}
        for dataset, label_ls in dict['dataset_based']:
            self.datset_c1c2[dataset] = [c1, c1+len(label_ls)]  # "AbdomenCT1K": [0, 4]
        
        self.batch_size = batch_size
        self.patch_size =patch_size
        
    def __len__(self):
        return len(self.lines)
    
    def _merge_modality(self, mod):
        if contains(mod, ['t1', 't2', 'mri', 'flair', 'dwi']):
            return 'mri'
        if contains(mod, 'ct'):
            return 'ct'
        if contains(mod, 'fundus'):
            return 'fundus'
        else:
            return mod
        
    def _pad_if_necessary(self, patch):
        # NOTE: depth must be pad to 96
        b, c, h, w, d = patch.shape
        t_h, t_w, t_d = self.patch_size
        pad_in_h = 0 if h >= t_h else t_h - h
        pad_in_w = 0 if w >= t_w else t_w - w
        pad_in_d = 0 if d >= t_d else t_d - d
        if pad_in_h + pad_in_w + pad_in_d > 0:
            pad = (0, pad_in_d, 0, pad_in_w, 0, pad_in_h)
            patch = F.pad(patch, pad, 'constant', 0)   # chwd
        return patch
        
    def __getitem__(self, idx):
        datum = self.lines[idx]
        sample_id = datum['renorm_image'].split('/')[-1][:-4]  # abcd/x.npy --> x
        
        # divide patches into batches
        patches = [torch.tensor(np.load(p)) for p in datum['patch_path']]
        batch_num = len(patches) // self.batch_size if len(patches) % self.batch_size == 0 else len(patches) // self.batch_size + 1
        batched_patches = []
        batched_y1y2_x1x2_z1z2 = []
        for i in range(batch_num):
            srt = i*self.batch_size
            end = min(i*self.batch_size+self.batch_size, len(patches))
            patch = torch.stack([patches[j] for j in range(srt, end)], dim=0)
            # NOTE: depth must be pad to 96
            patch = self._pad_if_necessary(patch)
            # for single-channel images, e.g. mri and ct, pad to 3
            # repeat sc image to mc
            if patch.shape[1] == 1:
                patch = repeat(patch, 'b c h w d -> b (c r) h w d', r=3)   
            batched_patches.append(patch) # b, *patch_size
            batched_y1y2_x1x2_z1z2.append([datum['patch_y1y2_x1x2_z1z2'][j] for j in range(srt, end)])
    
        # load gt segmentations
        c1, c2 = self.datset_c1c2[datum['dataset']]
        c,h,w,d = datum['chwd']
        labels = [datum['label'][i] for i in datum['visible_label_idx']] # laryngeal cancer or hypopharyngeal cancer
        mask_paths = [f"{datum['renorm_segmentation_dir']}/{label}.npy" for label in labels] # /remote-home/share/SAM/processed_files/Challenge_4C2021/segmentation/27/laryngeal cancer or hypopharyngeal cancer.npy
        y1x1z1_y2x2z2_ls = [datum['renorm_y1x1z1_y2x2z2'][i] for i in datum['visible_label_idx']]
        
        mc_mask = []
        for mask_path, y1x1z1_y2x2z2 in zip(mask_paths, y1x1z1_y2x2z2_ls):
            mask = torch.zeros((h, w, d))
            # not empty, load and embed non-empty cropped_volume
            if y1x1z1_y2x2z2 != False:
                y1, x1, z1, y2, x2, z2 = y1x1z1_y2x2z2
                try:
                    mask[y1:y2, x1:x2, z1:z2] = torch.tensor(np.load(mask_path))
                except:
                    print(mask_path)
            mc_mask.append(mask.float())
        mc_mask = torch.stack(mc_mask, dim=0)   # n h w d

        return {
            'dataset_name':datum['dataset'],
            'sample_id':sample_id, 
            'batched_patches':batched_patches, 
            'batched_y1y2_x1x2_z1z2':batched_y1y2_x1x2_z1z2, 
            'c1_c2':[c1, c2],
            'gt_segmentation':mc_mask,
            'labels':labels
            }
        
def collate_splited_images_and_labels(data):
    return data[0]
    