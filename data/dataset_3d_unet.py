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
from monai.transforms import (
    Compose,
    RandShiftIntensityd,
    RandRotate90d,
    RandZoomd,
)

def contains(text, key):
    if isinstance(key, str):
        return key in text
    elif isinstance(key, list):
        for k in key:
            if k in text:
                return True
        return False         
    
class Med_SAM_Dataset(Dataset):
    def __init__(self, 
                 jsonl_file, 
                 label_json,
                 dataset_config,
                 crop_size=[256,256,96], 
                 allow_repeat=True):
        """
        Assemble 46 segmentation datasets
        
        Args:
            json_file (_type_): a jsonl contains all train sample information
            crop_size (int, optional): _description_. Defaults to [256,256,96].
            max_queries (int, optional): _description_. Defaults to 16.
            dataset_config (str, optional): a path to config file, defining the sampling, loading parameters of each dataset etc
            allow_repeat (bool, optional): sample for multiply times to accelerate convergency. Defaults to True.
        """
        # data processing
        self.crop_size = crop_size
        
        with open(label_json, 'r') as f:
            dict = json.load(f)
        c1 = 0
        self.datset_c1c2 = {}
        for dataset, label_ls in dict['dataset_based'].items():
            self.datset_c1c2[dataset] = [c1, c1+len(label_ls)]  # "AbdomenCT1K": [0, 4]
            c1 += len(label_ls)
        
        # load data info
        with open(dataset_config, 'r') as f:
            self.dataset_config = json.load(f)
        
        self.jsonl_file = jsonl_file
        with open(self.jsonl_file, 'r') as f:
            lines = f.readlines()
        lines = [json.loads(line) for line in lines]
        
        local_num = 0
        self.data_2d = []    # complete list of data
        self.sample_weight_2d = []   # and their sampling weight
        self.data_3d = []
        self.sample_weight_3d = []
        self.datasets = set()
        count_2d_repeat = 0
        count_3d_repeat = 0
        for sample in lines:
            
            # if train on dbcloud, check the local copy availablity
            path = sample['renorm_image']
            if '/remote-home/share/SAM' in path:
                local_img_path_202 = path.replace('/remote-home/share/SAM', '/remote-home/share/data202/172.16.11.202/SAM')
                if os.path.exists(local_img_path_202):
                    sample['renorm_image'] = local_img_path_202
                    local_num += 1
                else:
                    local_img_path_118 = path.replace('/remote-home/share/SAM', '/remote-home/share/data118/SAM')
                    if os.path.exists(local_img_path_118):
                        sample['renorm_image'] = local_img_path_118
                        local_num += 1
                        
            path = sample['renorm_segmentation_dir']
            if '/remote-home/share/SAM' in path:
                local_img_path_202 = path.replace('/remote-home/share/SAM', '/remote-home/share/data202/172.16.11.202/SAM')
                if os.path.exists(local_img_path_202):
                    sample['renorm_segmentation_dir'] = local_img_path_202
                    local_num += 1
                else:
                    local_img_path_118 = path.replace('/remote-home/share/SAM', '/remote-home/share/data118/SAM')
                    if os.path.exists(local_img_path_118):
                        sample['renorm_segmentation_dir'] = local_img_path_118
                        local_num += 1
            
            # sampling weight
            self.datasets.add(sample['dataset'])
            weight = self.dataset_config[sample['dataset']]['sampling_weight'] # weigh to sample this sample
            # repeat
            query_repeat_times = 1
            # repeat for volume size
            _, h, w, d = sample['chwd']
            h_repeat_times = (h / crop_size[0])
            w_repeat_times = (w / crop_size[1])
            d_repeat_times = (d / crop_size[2])
            size_repeat_times = h_repeat_times * w_repeat_times * d_repeat_times
            if size_repeat_times < 1 or not allow_repeat:
                size_repeat_times = 1
            # repeat
            repeat_times = round(size_repeat_times * query_repeat_times)  # e.g. 1.5 * 2.5 = 3.75 --> 4
            if sample['is_3D']=='3D':
                for i in range(repeat_times):
                    self.data_3d.append(sample)
                    self.sample_weight_3d.append(weight)
                count_3d_repeat += (repeat_times - 1)
            elif sample['is_3D']=='2D':
                for i in range(repeat_times):
                    self.data_2d.append(sample)
                    self.sample_weight_2d.append(weight)
                count_2d_repeat += (repeat_times - 1)
            else:
                raise ValueError(f"data type {sample['is_3D']} is neither 2D or 3D")
            
        # determine sample weight and num
        self.num_2d = round(sum(self.sample_weight_2d)) # num of 3d and 2d samples must be fixed throughout epochs or the sampler wont work properly
        self.num_3d = round(sum(self.sample_weight_3d))
        self.data_split = {'2d':[0, self.num_2d], '3d':[self.num_2d, self.num_2d+self.num_3d]}
        
        if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
            print(f'** DATASET ** {len(self.data_2d)-count_2d_repeat} unique 2D samples are loaded, {count_2d_repeat} samples are repeated; {self.num_2d} samples will be sampled in an epoch')
            print(f'** DATASET ** {len(self.data_3d)-count_3d_repeat} unique 3D samples are loaded, {count_3d_repeat} samples are repeated; {self.num_3d} samples will be sampled in an epoch, {local_num/2} samples are in local')  
            print(f'** DATASET ** In total {len(self.datasets)} datasets.\n')
            print(f'** DATASET ** Configure for each dataset : {len(self.dataset_config)}')
            
        # sample for the first epoch
        self.sample_for_an_epoch()
        
        # data augmentation (tailor for each dataset)
        self.augmentator = {}
        for dataset in self.datasets:
            config = self.dataset_config[dataset]['augmentation']
            aug_ls = []
            if 'RandZoom' in config:
                aug_ls.append(RandZoomd(
                    keys=["image", "label"], 
                    mode=['area', 'nearest'],
                    min_zoom=config['RandZoom']['min_zoom'],
                    max_zoom=config['RandZoom']['max_zoom'],
                    prob=config['RandZoom']['prob'],
                    ))
            if 'RandRotate90' in config:
                aug_ls.append(RandRotate90d(
                    keys=["image", "label"], 
                    max_k=config['RandRotate90']['max_k'],
                    prob=config['RandRotate90']['prob'],
                    ))
            if 'RandIntensityShift' in config:
                aug_ls.append(RandShiftIntensityd(
                    keys=["image"], 
                    offsets=config['RandIntensityShift']['offsets'],
                    prob=config['RandIntensityShift']['prob'],
                    ))
            if len(aug_ls) > 0:
                self.augmentator[dataset] = Compose(aug_ls)
                
    def __len__(self):
        return len(self.datasets_samples)
        
    def adjust_dataset_config(self, f_path):
        """
        Adjust the config of datasets
        """
        if os.path.exists(f_path):
            with open(f_path, 'r') as f:
                self.dataset_config = json.load(f)
        else:
            return
        
        print(f'** DATASET ** Adjust config for each dataset : {f_path}')
                
        self.sample_weight_2d = []   # sampling weight
        self.sample_weight_3d = []
        
        for sample in self.data_3d:
            dataset = sample['dataset']
            self.sample_weight_3d.append(self.dataset_config[dataset]['sampling_weight'])
        for sample in self.data_2d:
            dataset = sample['dataset']
            self.sample_weight_2d.append(self.dataset_config[dataset]['sampling_weight'])
            
        # Regenerate augmentation for each dataset
            
        self.augmentator = {}
        for dataset in self.datasets:
            config = self.dataset_config[dataset]['augmentation']
            aug_ls = []
            if 'RandZoom' in config:
                aug_ls.append(RandZoomd(
                    keys=["image", "label"], 
                    mode=['area', 'nearest'],
                    min_zoom=config['RandZoom']['min_zoom'],
                    max_zoom=config['RandZoom']['max_zoom'],
                    prob=config['RandZoom']['prob'],
                    ))
            if 'RandRotate90' in config:
                aug_ls.append(RandRotate90d(
                    keys=["image", "label"], 
                    max_k=config['RandRotate90']['max_k'],
                    prob=config['RandRotate90']['prob'],
                    ))
            if 'RandIntensityShift' in config:
                aug_ls.append(RandShiftIntensityd(
                    keys=["image"], 
                    offsets=config['RandIntensityShift']['offsets'],
                    prob=config['RandIntensityShift']['prob'],
                    ))
            if len(aug_ls) > 0:
                self.augmentator[dataset] = Compose(aug_ls)
        
    def sample_for_an_epoch(self):
        """
        Some dataset may NOT be fully sampled (e.g. those converged ones, refer to self.dataset_sample_weights)
        Therefore, resampling from them in every epoch to ensure randomness is necessary
        """
        if self.num_2d > 0:
            sample_2d = random.choices(self.data_2d, weights=self.sample_weight_2d, k=self.num_2d)
        else:
            sample_2d = []
        
        if self.num_3d > 0:    
            sample_3d = random.choices(self.data_3d, weights=self.sample_weight_3d, k=self.num_3d)
        else:
            sample_3d = []
            
        self.datasets_samples = sample_2d + sample_3d
    
    def _merge_modality(self, mod):
        if contains(mod, ['t1', 't2', 'mri', 'flair', 'dwi']):
            return 'mri'
        if contains(mod, 'ct'):
            return 'ct'
        if contains(mod, 'fundus'):
            return 'fundus'
        else:
            return mod
    
    def _pad_if_necessary(self, image, mask):
        # image size >= crop size 
        c, h, w, d = image.shape
        croph, cropw, cropd = self.crop_size
        pad_in_h = 0 if h >= croph else croph - h
        pad_in_w = 0 if w >= cropw else cropw - w
        pad_in_d = 0 if d >= cropd else cropd - d
        if pad_in_h + pad_in_w + pad_in_d > 0:
            pad = (0, pad_in_d, 0, pad_in_w, 0, pad_in_h)
            mask = F.pad(mask, pad, 'constant', 0)   # nhwd
            image = F.pad(image, pad, 'constant', 0)   # chwd
        
        return image, mask
    
    def _crop(self, image, mask, is_pos, label_based_crop_prob):
        # c h w d & n h w d
        _, imgh, imgw, imgd = image.shape
        croph, cropw, cropd = self.crop_size
        if (imgh - croph) > 0 or (imgw - cropw) > 0 or (imgd - cropd) > 0:
            # need crop
            if not (True in is_pos):
                image, mask, y1x1z1_y2x2z2 = self._random_crop(image, mask)
            else:
                image, mask, y1x1z1_y2x2z2 = self._roi_crop(image, mask, is_pos, label_based_crop_prob)
        else:
            y1x1z1_y2x2z2 = [0, 0, 0, imgh, imgw, imgd]
                
        return image, mask, y1x1z1_y2x2z2
    
    def _roi_crop(self, image, mask, is_pos, label_based_crop_prob):
        # c h w d & n h w d
        _, imgh, imgw, imgd = image.shape
        croph, cropw, cropd = self.crop_size
        
        if random.random() < label_based_crop_prob:
            # find a pos label and crop based on it (ensure at least one pos label before roi crop
            pos_label_idx_ls = [i for i, t_or_f in enumerate(is_pos) if t_or_f]
            pos_label_idx = random.sample(pos_label_idx_ls, 1)[0]
            mask_to_select = mask[pos_label_idx, :, :, :]  # h w d 
        else:
            # crop based on all labels
            mask_to_select = torch.sum(mask, dim=0)   # h w d
        
        # select a voxel
        voxels_foreground = torch.nonzero(mask_to_select, as_tuple=True)   # (tensor(...), tensor(...), tensor(...))
        selected_index = random.randint(0, voxels_foreground[0].shape[0]-1)
        selected_voxel = (voxels_foreground[0][selected_index].item(), voxels_foreground[1][selected_index].item(), voxels_foreground[2][selected_index].item())
        
        # check the boundary
        if selected_voxel[0] - croph // 2 > 0:
            start_y = selected_voxel[0] - croph // 2
            if start_y + croph < imgh:
                end_y = start_y + croph
            else:
                end_y = imgh
                start_y = imgh-croph
        else:
            start_y = 0
            end_y = croph
            
        if selected_voxel[1] - cropw // 2 > 0:
            start_x = selected_voxel[1] - cropw // 2
            if start_x + cropw < imgw:
                end_x = start_x + cropw
            else:
                end_x = imgw
                start_x = imgw-cropw
        else:
            start_x = 0
            end_x = cropw

        if selected_voxel[2] - cropd // 2 > 0:
            start_z = selected_voxel[2] - cropd // 2
            if start_z + cropd < imgd:
                end_z = start_z + cropd
            else:
                end_z = imgd
                start_z = imgd-cropd
        else:
            start_z = 0
            end_z = cropd  
        
        # randomly shift the crop (must contain the selected voxel
        y_left_space = min(start_y - 0, end_y - selected_voxel[0])
        y_right_space = min(imgh - end_y, selected_voxel[0] - start_y)
        y_adjust = random.randint(-1 * y_left_space, y_right_space)
        start_y += y_adjust
        end_y += y_adjust
        
        x_left_space  = min(start_x-0, end_x-selected_voxel[1])
        x_right_space = min(imgw-end_x, selected_voxel[1]-start_x)
        x_adjust = random.randint(-1*x_left_space, x_right_space)
        start_x += x_adjust
        end_x += x_adjust

        z_left_space = min(start_z - 0, end_z - selected_voxel[2])
        z_right_space = min(imgd - end_z, selected_voxel[2] - start_z)
        z_adjust = random.randint(-1 * z_left_space, z_right_space)
        start_z += z_adjust
        end_z += z_adjust
        
        # crop
        crop_image = image[:, start_y:end_y, start_x:end_x, start_z:end_z]
        crop_mask = mask[:, start_y:end_y, start_x:end_x, start_z:end_z]

        return crop_image, crop_mask, [start_y, start_x, start_z, end_y, end_x, end_z]
    
    def _random_crop(self, image, mask):
        # c h w d & n h w d
        _, imgh, imgw, imgd = image.shape
        croph, cropw, cropd = self.crop_size
        # 
        start_y = random.randint(0, imgh - croph)
        end_y = start_y + croph
        start_x = random.randint(0, imgw - cropw)
        end_x = start_x + cropw
        start_z = random.randint(0, imgd - cropd)
        end_z = start_z + cropd
        #
        crop_image = image[:, start_y:end_y, start_x:end_x, start_z:end_z]
        crop_mask = mask[:, start_y:end_y, start_x:end_x, start_z:end_z]
        
        return crop_image, crop_mask, [start_y, start_x, start_z, end_y, end_x, end_z]
    
    def _select_pos_labels(self, visible_label_index_ls, y1x1z1_y2x2z2_ls, neg_label_ratio_threshold):
        """
        尽可能多采positive的label同时控制negative的数量不能超过positive的一定比例
        """
        # count
        pos_label_index_ls = []
        neg_label_index_ls = []
        for i in visible_label_index_ls:
            if y1x1z1_y2x2z2_ls[i]:
                pos_label_index_ls.append(i)
            else:
                neg_label_index_ls.append(i)
        pos_num = len(pos_label_index_ls)
        neg_num = len(neg_label_index_ls)
        
        if pos_num == 0:
            # degrad to random sample
            sample_num = min(self.max_queries, len(visible_label_index_ls))
            chosen_label_index_ls = random.sample(visible_label_index_ls, sample_num)
            is_pos = [False] * sample_num
            return chosen_label_index_ls, is_pos
        
        # indicate each sample is pos or neg
        is_pos = []
        
        if pos_num <= self.max_queries:
            # all pos labels are included, then sample some neg labels
            chosen_label_index_ls = pos_label_index_ls 
            is_pos += [True] * pos_num
            max_neg_num = int(neg_label_ratio_threshold * pos_num)    # neg label num < (pos label num) * x%
            left_pos_num = min(self.max_queries-pos_num, max_neg_num)   # neg label num < self.max_queries-pos_num
            if neg_num <= left_pos_num:
                # neg are all sampled
                chosen_label_index_ls += neg_label_index_ls
                is_pos += [False] * neg_num
            else:
                # neg are sampled to control the ratio and max label num
                chosen_label_index_ls += random.sample(neg_label_index_ls, left_pos_num)
                is_pos += [False] * left_pos_num
        else:
            # no neg labels are sampled
            chosen_label_index_ls = random.sample(pos_label_index_ls, self.max_queries)
            is_pos += [True] * self.max_queries

        return chosen_label_index_ls, is_pos
    
    def _load_mask(self, datum):
        """
        加载segmentation mask
        Args:
            datum (dict): sample info (a line from jsonl file

        Returns:
            mc_mask: (n, h, w, d)
        """
        _, h, w, d = datum['chwd']
        mask_paths = [f"{datum['renorm_segmentation_dir']}/{label}.npy" for label in datum['label']] # /remote-home/share/SAM/processed_files/Challenge_4C2021/segmentation/27/laryngeal cancer or hypopharyngeal cancer.npy
        y1x1z1_y2x2z2_ls = datum['renorm_y1x1z1_y2x2z2']
        mc_mask = []
        is_pos = []
        for mask_path, y1x1z1_y2x2z2 in zip(mask_paths, y1x1z1_y2x2z2_ls):
            mask = torch.zeros((h, w, d), dtype=torch.bool)
            # not empty, load and embed non-empty cropped_volume
            if y1x1z1_y2x2z2 != False:
                y1, x1, z1, y2, x2, z2 = y1x1z1_y2x2z2
                mask[y1:y2, x1:x2, z1:z2] = torch.tensor(np.load(mask_path))
                is_pos.append(True)
            else:
                is_pos.append(False)
            mc_mask.append(mask)
        mc_mask = torch.stack(mc_mask, dim=0)   # n h w d
        
        return mc_mask, is_pos
    
    def _load_image(self, datum):
        image = torch.tensor(np.load(datum['renorm_image']))

        return image # chwd
        
    def __getitem__(self, idx):
        while True:
            try: 
                sample = self.datasets_samples[idx]
                
                # load image
                image = self._load_image(sample)
                
                # load masks
                mask, is_pos = self._load_mask(sample)
                c1, c2 = self.datset_c1c2[sample['dataset']]
                
                _, H, W, D = image.shape
                N, mH, mW, mD = mask.shape
                assert H == mH and W == mW and D == mD, f'image shape {H, W, D} inconsistent with mask shape {mH, mW, mD}'
                assert N == len(is_pos) and N == c2 - c1, f'mask shape {H, W, D} inconsistent with [c1,c2) : [{c1, c2})'
                
                # Pad image and mask
                image, mask = self._pad_if_necessary(image, mask)
                
                # Crop image and mask
                label_based_crop_prob = self.dataset_config[sample['dataset']]['label_based_crop_prob']
                image, mask, y1x1z1_y2x2z2 = self._crop(image, mask, is_pos, label_based_crop_prob)
                
                # augmentation if needed
                if sample['dataset'] in self.augmentator:
                    data_dict = {'image': image, 'label': mask}
                    aug_data_dict = self.augmentator[sample['dataset']](data_dict)
                    image, mask = aug_data_dict['image'], aug_data_dict['label']
                
                break
            except SystemExit:
                exit()
            except:
                # 检查数据load的bug
                traceback_info = traceback.format_exc()
                print(f'*** {sample["dataset"]} *** image : {sample["renorm_image"]} *** mask : {sample["renorm_segmentation_dir"]} ***\n')
                print(traceback_info)
                """with open(f'/remote-home/share/SAM/dataloading_error/{sample["dataset"]}.txt', 'a') as f:
                    f.write(f'*** {sample["dataset"]} *** image : {sample["renorm_image"]} *** mask : {sample["renorm_segmentation_dir"]} ***\n')
                    f.write(traceback_info + '\n\n')"""
                    
                idx = random.randint(0, len(self.datasets_samples)-1)

        return {'image':image, 'mask':mask, 'label':sample['label'], 'image_path':sample['renorm_image'], 'mask_path':sample['renorm_segmentation_dir'], 'dataset':sample['dataset'], 'c1_c2':[c1, c2], 'y1x1z1_y2x2z2':y1x1z1_y2x2z2}
    
if __name__ == '__main__':
    import nibabel as nib
    
    dataset = Med_SAM_Dataset(
        '/remote-home/share/SAM/trainsets_v3/TotalSegmentator_Organs(1).jsonl',
        crop_size=[256,256,96], 
        patch_size=32,
        max_queries=16, 
        label_based_crop_prob=0.5,
        pos_label_first_prob=1.0,
        neg_label_ratio_threshold=0.25,
        dataset_sample_weights='/remote-home/zihengzhao/Knowledge-Enhanced-Medical-Segmentation/medical-universal-segmentation/data/sampling_weight_config/the_seven.json',
        load_and_save='/remote-home/zihengzhao/Knowledge-Enhanced-Medical-Segmentation/medical-universal-segmentation/data/load_save_config/the_seven.json',
        allow_repeat=False
    )
    
    for idx in range(10):
        Path(f'/remote-home/zihengzhao/Knowledge-Enhanced-Medical-Segmentation/medical-universal-segmentation/debug_dataset_3d/TotalSegmentator_Organs(1)/sample_{idx}').mkdir(exist_ok=True, parents=True)
        
        print(f'\n\n** SAMPLE {idx} **\n\n')
        sample = dataset[0]
    
        # 保存label信息
        # 保存patch信息
        image = sample['image'].numpy()
        y1, x1, z1, y2, x2, z2 = sample['y1x1z1_y2x2z2']
        imgobj = nib.nifti2.Nifti1Image(image[0, :, :, :], np.eye(4))
        nib.save(imgobj, f'/remote-home/zihengzhao/Knowledge-Enhanced-Medical-Segmentation/medical-universal-segmentation/debug_dataset_3d/TotalSegmentator_Organs(1)/sample_{idx}/{y1}:{y2}_{x1}:{x2}_{z1}:{z2}_image.nii.gz')
        
        tl = sample['text']
        mask = sample['mask'].numpy()
        c, h, w, d = image.shape
        mc_mask = np.zeros((h, w, d))
        for i, t in enumerate(tl):
            t = t.split('Anatomy Name:')[-1]
            mc_mask += mask[i, :, :, :] * (i+1)
            segobj = nib.nifti2.Nifti1Image(mask[i, :, :, :], np.eye(4))
            nib.save(segobj, f'/remote-home/zihengzhao/Knowledge-Enhanced-Medical-Segmentation/medical-universal-segmentation/debug_dataset_3d/TotalSegmentator_Organs(1)/sample_{idx}/{t}.nii.gz')
        segobj = nib.nifti2.Nifti1Image(mc_mask[:, :, :], np.eye(4))
        nib.save(segobj, f'/remote-home/zihengzhao/Knowledge-Enhanced-Medical-Segmentation/medical-universal-segmentation/debug_dataset_3d/TotalSegmentator_Organs(1)/sample_{idx}/labels.nii.gz')
        
        with open(f'/remote-home/zihengzhao/Knowledge-Enhanced-Medical-Segmentation/medical-universal-segmentation/debug_dataset_3d/TotalSegmentator_Organs(1)/sample_{idx}/labels.txt', 'a') as f:
            for i, t in enumerate(tl):
                t = t.split('Anatomy Name:')[-1]
                f.write(f'{i} -- {t}\n')            