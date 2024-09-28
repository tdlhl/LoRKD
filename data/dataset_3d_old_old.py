import os
import random
import math

from einops import rearrange, repeat, reduce
import json
import numpy as np
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
    RandGaussianNoised,
    RandGaussianSharpend,
    RandScaleIntensityd,
    #RandScaleIntensityFixedMeand,
    #RandSimulateLowResolutiond,
    RandAdjustContrastd
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
                 dataset_config,
                 crop_size=[288,288,96], 
                 max_queries=16, 
                 allow_repeat=True):
        """
        Assemble segmentation datasets
        
        Args:
            json_file (_type_): a jsonl contains all train sample information
            crop_size (int, optional): _description_. Defaults to [288,288,96].
            max_queries (int, optional): _description_. Defaults to 32.
            dataset_config (str, optional): a path to config file, defining the sampling, loading parameters of each dataset etc
            allow_repeat (bool, optional): sample for multiply times to accelerate convergency. Defaults to True.
        """
        self.crop_size = crop_size
        self.max_queries = max_queries
        
        # load data configs
        with open(dataset_config, 'r') as f:
            self.dataset_config = json.load(f)
        
        # load 
        self.jsonl_file = jsonl_file
        with open(self.jsonl_file, 'r') as f:
            lines = f.readlines()
        lines = [json.loads(line) for line in lines]
        
        # statistics the size of each dataset and the repeated times and the sampled times within a log interval
        datasets_dist = [l['dataset'] for l in lines]
        self.datasets = set(datasets_dist)
        self.datasets_size = {}
        self.datasets_repeat_times = {}
        for dataset in self.datasets:
            self.datasets_size[dataset] = datasets_dist.count(dataset)
            self.datasets_repeat_times[dataset] = 0       
    
        local_num = 0
        self.data_3d = []   # list of data samples
        self.sample_weight_3d = []  # and their sampling weight
        count_3d_repeat = 0
        """
        self.data_2d = []
        self.sample_weight_2d = []
        count_2d_repeat = 0
        """
        
        for sample in lines:
            
            # if train on dbcloud, check the local copy availablity
            path = sample['renorm_image']
            if '/remote-home/share/share/SAM' in path:
                raise NotImplementedError('Not determine the local path on 202')
                local_img_path_202 = path.replace('/remote-home/share/share/SAM', '/remote-home/share/data-H800-202/zihengzhao/SAM')
                local_img_path_118 = path.replace('/remote-home/share/share/SAM', '/remote-home/share/data118/SAM')
                local_img_path_116 = path.replace('/remote-home/share/share/SAM', '/remote-home/share/data116/SAM')
                if os.path.exists(local_img_path_202):
                    sample['renorm_image'] = local_img_path_202
                    local_num += 1
                elif os.path.exists(local_img_path_118):
                    sample['renorm_image'] = local_img_path_118
                    local_num += 1
                elif os.path.exists(local_img_path_116):
                    sample['renorm_image'] = local_img_path_116
                    local_num += 1
                else:
                    print(f'Non-Local: {path}')
                        
            path = sample['renorm_segmentation_dir']
            if '/remote-home/share/share/SAM' in path:
                raise NotImplementedError('Not determine the local path on 202')
                local_seg_path_202 = path.replace('/remote-home/share/share/SAM', '/remote-home/share/data-H800-202/zihengzhao/SAM')
                local_seg_path_118 = path.replace('/remote-home/share/share/SAM', '/remote-home/share/data118/SAM')
                local_seg_path_116 = path.replace('/remote-home/share/share/SAM', '/remote-home/share/data116/SAM')
                if os.path.exists(local_seg_path_202):
                    sample['renorm_segmentation_dir'] = local_seg_path_202
                    local_num += 1
                elif os.path.exists(local_seg_path_118):
                    sample['renorm_segmentation_dir'] = local_seg_path_118
                    local_num += 1
                elif os.path.exists(local_seg_path_116):
                    sample['renorm_segmentation_dir'] = local_seg_path_116
                    local_num += 1
                else:
                    print(f'Non-Local: {path}')
            
            # sampling weight : inverse to square root of dataset size
            size = self.datasets_size[sample['dataset']]
            weight = 1 / (math.sqrt(size))
            # sampling weight : allow manual adjustment in data config file
            weight = weight * self.dataset_config[sample['dataset']]['sampling_weight']
            
            # repeat times for label num
            query_repeat_times = max(1, (len(sample['label']) / max_queries))
            # repeat times for roi size
            if sample['roi_y1x1z1_y2x2z2']:
                y1, x1, z1, y2, x2, z2 = sample['roi_y1x1z1_y2x2z2']
                h_repeat_times = max(1, ((y2-y1) / crop_size[0]))
                w_repeat_times = max(1, ((x2-x1) / crop_size[1]))
                d_repeat_times = max(1, ((z2-z1) / crop_size[2]))
                size_repeat_times = h_repeat_times * w_repeat_times * d_repeat_times
            else:
                size_repeat_times = 1
                
            # not repeat
            if not allow_repeat:
                size_repeat_times = query_repeat_times = 1
            # allow repeat
            repeat_times = round(size_repeat_times * query_repeat_times)  # e.g. 1.5 * 2.5 = 3.75 --> 4
            if 'is_3D' not in sample or sample['is_3D']=='3D':
                for i in range(round(repeat_times)):
                    self.data_3d.append(sample)
                    self.sample_weight_3d.append(weight)
                count_3d_repeat += (repeat_times - 1)
                self.datasets_repeat_times[sample['dataset']] += (repeat_times - 1)
            elif sample['is_3D']=='2D':
                for i in range(round(repeat_times)):
                    self.data_2d.append(sample)
                    self.sample_weight_2d.append(weight)
                count_2d_repeat += (repeat_times - 1)
                self.datasets_repeat_times[sample['dataset']] += (repeat_times - 1)
            else:
                raise ValueError(f"data type {sample['is_3D']} is neither 2D or 3D")
            
        """
        # determine sample weight and num
        self.num_2d = 0
        self.num_3d = len(self.data_3d)
        self.data_split = {'2d':[0, self.num_2d], '3d':[self.num_2d, self.num_2d+self.num_3d]}
        """
        
        if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
            # print(f'** DATASET ** {len(self.data_2d)-count_2d_repeat} unique 2D samples are loaded, {count_2d_repeat} samples are repeated; {self.num_2d} samples will be sampled in an epoch')
            print(f'** DATASET ** {len(lines)} unique 3D samples are loaded, {count_3d_repeat} samples are repeated; {local_num/2} samples are in local')  
            print(f'** DATASET ** In total {len(self.datasets)} datasets.\n')
            # print(f'** DATASET ** Configure for each dataset : {len(self.dataset_config)}')
            print(f'** DATASET ** Size, Repeated Times and Repeat/Size Ratio for each dataset:\n')
            for k,repeated_times in self.datasets_repeat_times.items():
                size = self.datasets_size[k]
                print(f'{k} : {size}/{repeated_times} = {repeated_times/size}')
        
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
                    )
                )
            if 'RandGaussianNoise' in config:
                aug_ls.append(
                    RandGaussianNoised(
                        keys=['image'],
                        prob=config['RandGaussianNoise']['prob'],
                        mean=config['RandGaussianNoise']['mean'],
                        std=0.1
                    )
                )
            if 'RandGaussianSharpen' in config:
                aug_ls.append(
                    RandGaussianSharpend(
                        keys=['image'],
                        prob=config['RandGaussianSharpen']['prob'],
                    )
                )
            if 'RandScaleIntensity' in config:
                aug_ls.append(
                    RandScaleIntensityd(
                        keys=['image'],
                        factors=config['RandScaleIntensity']['factors'],
                        prob=config['RandScaleIntensity']['prob']
                    )
                )
            """if 'RandScaleIntensityFixedMean' in config:
                aug_ls.append(
                    RandScaleIntensityFixedMeand(
                        keys=['image'],
                        factors=config['RandScaleIntensityFixedMean']['factors'],
                        prob=config['RandScaleIntensityFixedMean']['prob']
                    )
                )
            if 'RandSimulateLowResolution' in config:
                aug_ls.append(
                    RandSimulateLowResolutiond(
                        keys=['image'],
                        prob=config['RandSimulateLowResolution']['prob']
                    )
                )"""
            if 'RandAdjustContrastInvert' in config:
                aug_ls.append(
                    RandAdjustContrastd(
                        keys=['image'],
                        #retain_stats=config['RandAdjustContrastInvert']['retain_stats'],
                        #invert_image=config['RandAdjustContrastInvert']['invert_image'],
                        gamma=config['RandAdjustContrastInvert']['gamma'],
                        prob=config['RandAdjustContrastInvert']['prob']
                    )
                )
            if 'RandAdjustContrast' in config:
                aug_ls.append(
                    RandAdjustContrastd(
                        keys=['image'],
                        #retain_stats=config['RandAdjustContrast']['retain_stats'],
                        #invert_image=config['RandAdjustContrast']['invert_image'],
                        gamma=config['RandAdjustContrast']['gamma'],
                        prob=config['RandAdjustContrast']['prob']
                    )
                )
            if len(aug_ls) > 0:
                self.augmentator[dataset] = Compose(aug_ls)
        
    def __len__(self):
        return 1000000000 # life long training ... (10e9)
        
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
            size = self.datasets_size[dataset]
            weight = 1 / (math.sqrt(size))
            self.sample_weight_3d.append(self.dataset_config[dataset]['sampling_weight'] * weight)
        for sample in self.data_2d:
            dataset = sample['dataset']
            size = self.datasets_size[dataset]
            weight = 1 / (math.sqrt(size))
            self.sample_weight_2d.append(self.dataset_config[dataset]['sampling_weight'] * weight)
            
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
                    )
                )
            if 'RandGaussianNoise' in config:
                aug_ls.append(
                    RandGaussianNoised(
                        keys=['image'],
                        prob=config['RandGaussianNoise']['prob'],
                        mean=config['RandGaussianNoise']['mean'],
                        std=0.1
                    )
                )
            if 'RandGaussianSharpen' in config:
                aug_ls.append(
                    RandGaussianSharpend(
                        keys=['image'],
                        prob=config['RandGaussianSharpen']['prob'],
                    )
                )
            if 'RandScaleIntensity' in config:
                aug_ls.append(
                    RandScaleIntensityd(
                        keys=['image'],
                        factors=config['RandScaleIntensity']['factors'],
                        prob=config['RandScaleIntensity']['prob']
                    )
                )
            """if 'RandScaleIntensityFixedMean' in config:
                aug_ls.append(
                    RandScaleIntensityFixedMeand(
                        keys=['image'],
                        factors=config['RandScaleIntensityFixedMean']['factors'],
                        prob=config['RandScaleIntensityFixedMean']['prob']
                    )
                )
            if 'RandSimulateLowResolution' in config:
                aug_ls.append(
                    RandSimulateLowResolutiond(
                        keys=['image'],
                        prob=config['RandSimulateLowResolution']['prob']
                    )
                )"""
            if 'RandAdjustContrastInvert' in config:
                aug_ls.append(
                    RandAdjustContrastd(
                        keys=['image'],
                        retain_stats=config['RandAdjustContrastInvert']['retain_stats'],
                        invert_image=config['RandAdjustContrastInvert']['invert_image'],
                        gamma=config['RandAdjustContrastInvert']['gamma'],
                        prob=config['RandAdjustContrastInvert']['prob']
                    )
                )
            if 'RandAdjustContrast' in config:
                aug_ls.append(
                    RandAdjustContrastd(
                        keys=['image'],
                        retain_stats=config['RandAdjustContrast']['retain_stats'],
                        invert_image=config['RandAdjustContrast']['invert_image'],
                        gamma=config['RandAdjustContrast']['gamma'],
                        prob=config['RandAdjustContrast']['prob']
                    )
                )
            if len(aug_ls) > 0:
                self.augmentator[dataset] = Compose(aug_ls)

    def _merge_modality(self, mod):
        if contains(mod, ['t1', 't2', 'mri', 'flair', 'dwi']):
            return 'mri'
        if contains(mod, 'ct'):
            return 'ct'
        if contains(mod, 'pet'):
            return 'pet'
        if contains(mod, ['us', 'ultrasound']):
            return 'us'
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
    
    def _crop(self, image, mask, is_pos, roi_crop_prob, label_based_crop_prob, uncenter_prob):
        # c h w d & n h w d
        _, imgh, imgw, imgd = image.shape
        croph, cropw, cropd = self.crop_size
        if (imgh - croph) > 0 or (imgw - cropw) > 0 or (imgd - cropd) > 0:
            # need crop
            if not (True in is_pos) or random.random() > roi_crop_prob:
                # no roi region
                image, mask, y1x1z1_y2x2z2 = self._random_crop(image, mask)
            else:
                # 100% roi crop
                image, mask, y1x1z1_y2x2z2 = self._roi_crop(image, mask, is_pos, label_based_crop_prob, uncenter_prob)
        else:
            y1x1z1_y2x2z2 = [0, 0, 0, imgh, imgw, imgd]
                
        return image, mask, y1x1z1_y2x2z2
    
    def _roi_crop(self, image, mask, is_pos, label_based_crop_prob, uncenter_prob):
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
        if random.random() < uncenter_prob:
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
    
    def _load_mask(self, datum, chosen_label_indexes):
        """
        按需加载segmentation mask
        Args:
            datum (dict): sample info (a line from jsonl file
            chosen_label_indexes (List[int]): N indexes of sampled labels

        Returns:
            mc_mask: (n, h, w, d)
            labels: list of N str
        """
        _, h, w, d = datum['chwd']
        labels = [datum['label'][i] for i in chosen_label_indexes] # laryngeal cancer or hypopharyngeal cancer
        mask_paths = [f"{datum['renorm_segmentation_dir']}/{label}.npy" for label in labels] # /remote-home/share/SAM/processed_files/Challenge_4C2021/segmentation/27/laryngeal cancer or hypopharyngeal cancer.npy
        y1x1z1_y2x2z2_ls = [datum['renorm_y1x1z1_y2x2z2'][i] for i in chosen_label_indexes] 
        mc_mask = []
        for mask_path, y1x1z1_y2x2z2 in zip(mask_paths, y1x1z1_y2x2z2_ls):
            mask = torch.zeros((h, w, d), dtype=torch.bool)
            # print('mask_path=', mask_path)
            # print('y1x1z1_y2x2z2=', y1x1z1_y2x2z2)
            # not empty, load and embed non-empty cropped_volume
            if y1x1z1_y2x2z2 != False:
                y1, x1, z1, y2, x2, z2 = y1x1z1_y2x2z2
                # print(y1, x1, z1, y2, x2, z2)
                mask[y1:y2, x1:x2, z1:z2] = torch.tensor(np.load(mask_path))
            mc_mask.append(mask)
        mc_mask = torch.stack(mc_mask, dim=0)   # n h w d
        
        return mc_mask, labels
    
    def _load_image(self, datum):
        # if the local copy exists
        # NOTE: Make sure the local copy consistent
        path = datum['renorm_image']
        image = torch.tensor(np.load(path))

        return image # chwd
    
    def get_size_and_repeat(self, dataset_name):
        return self.datasets_size[dataset_name], self.datasets_repeat_times[dataset_name]

    def convert_to_labels(self, batch):
        # 定义转换规则
        #0-Abdomen, 1-Brain 2-H&N 3-LL 4-Pelvis 5-Spine 6-Thorax 7-UL
        #TotalSegmentator_Organs，TotalSegmentator_Muscles里面有好多部位，
        #现在准备让它直接过通用的backbone，然后3全在这两个里面
        dataset_to_label = {
            "CTPelvic1K" :  {5, 4} ,
            "SegRap2023_Task1" :  {5, 2, 1} ,
            "PDDCA" :  {2, 1} ,
            "SegTHOR" :  {0, 2, 6} ,
            "CT_ORG" :  {4, 8, 0, 2, 6} ,
            "AMOS22_MRI" :  {4, 0, 6} ,
            "AMOS22_CT" :  {0, 2, 4} ,
            "FLARE22" :  {0, 2} ,
            "WORD" :  {0, 2, 7, 4} ,
            "TotalSegmentator_Organs" :  {0, 4, 2, 6} ,
            "TotalSegmentator_Cardiac" :  {4, 0, 6} ,
            "TotalSegmentator_Muscles" :  {3, 4, 7, 2, 6} ,
            "BrainPTM" :  {2, 1},
            "HAN_Seg" :  {5, 2, 1} ,
            "LUNA16" :  {2, 6} ,
            #
            "SegRap2023_Task2" :  2 ,
            "LNDb" :  6 ,
            "FUMPE" :  6 ,
            "Pancreas_CT" :  0 ,
            "VerSe" :  5 ,
            "SEGA" :  0 ,
            "KiPA22" :  0 ,
            "ATLAS" :  0 ,
            "KiTS23" :  0 ,
            "Instance22" :  1 ,
            "MyoPS2020" :  6 ,
            "ATLASR2" :  1 ,
            "LAScarQS22_Task2" :  6 ,
            "CrossMoDA2021" :  2 ,
            "LAScarQS22_Task1" :  6 ,
            "MM_WHS_MRI" :  6 ,
            "MM_WHS_CT" :  6 ,
            "CMRxMotion" :  6 ,
            "PARSE2022" :  6 ,
            "FeTA2022" :  1 ,
            "AbdomenCT1K" :  0 ,
            "TotalSegmentator_Ribs" :  6 ,
            "TotalSegmentator_Vertebrae" :  5 ,
            "NSCLC" :  6 ,
            "CHAOS_MRI" :  0 ,
            "ISLES2022" :  1 ,
            "ACDC" :  6 ,
            "MRSpineSeg" :  5 ,
            "PROMISE12" :  4 ,
            "SKI10" :  7 ,
            "SLIVER07" :  0 ,
            "WMH_Segmentation_Challenge" :  1 ,
            "Brain_Atlas" :  1 ,
            "Couinaud_Liver" :  0 ,
        }

        # if 'TotalSegmentator' in batch['dataset']:
        
        # 转换batch['dataset']中的每个元素
        task_labels = []
        for item in batch['dataset']:
            task_label = dataset_to_label[item]
            task_label_one_hot = F.one_hot(torch.tensor(task_label), num_classes = self.args.num_tasks)
            task_labels.append(task_label_one_hot)
        
        task_labels_tensor = torch.stack(task_labels, dim=0)
        return task_labels_tensor
        
    def __getitem__(self, idx):
        while True:
            try: 
                sample = random.choices(self.data_3d, weights=self.sample_weight_3d)[0]
                
                # load image
                image = self._load_image(sample)
                
                # choose label/queries（up to max_queries
                visible_label_index_ls = [i for i in range(len(sample['label']))]# sample['visible_label_idx'] # available labels, their idx in channel-dim   [0, 1, ...]
                pos_label_first_prob = self.dataset_config[sample['dataset']]['pos_label_first_prob']
                neg_label_ratio_threshold = self.dataset_config[sample['dataset']]['neg_label_ratio_threshold']
                if random.random() < pos_label_first_prob:
                    # sample pos labels as many as possible (could be no pos?
                    chosen_label_index_ls, is_pos = self._select_pos_labels(visible_label_index_ls, sample['renorm_y1x1z1_y2x2z2'], neg_label_ratio_threshold)
                else:
                    chosen_label_index_ls = random.sample(visible_label_index_ls, min(self.max_queries, len(visible_label_index_ls)))
                    is_pos = [True if sample['renorm_y1x1z1_y2x2z2'][i] else False for i in chosen_label_index_ls]
                
                # print('chosen_label_index_ls=', chosen_label_index_ls)
                # print('visible_label_index_ls=', visible_label_index_ls)
                # load masks based on chosen labels
                # print('sample=', sample)
                # print('chosen_label_index_ls=', chosen_label_index_ls)
                mask, label_ls = self._load_mask(sample, chosen_label_index_ls)
                # raise ValueError("value must be an integer")
                
                # simple check
                _, H, W, D = image.shape
                N, mH, mW, mD = mask.shape
                assert H == mH and W == mW and D == mD, f'image shape {H, W, D} inconsistent with mask shape {mH, mW, mD}'
                assert N == len(label_ls), f'query num {len(label_ls)} inconsistent with gt mask channels {N}'
                
                # merge modality, e.g. t1 -> mri
                modality = sample['modality']
                modality = self._merge_modality(modality.lower())
                    
                # text prompt = modality + label name
                label_ls = [label.lower() for label in label_ls]
                
                # pad image and mask
                image, mask = self._pad_if_necessary(image, mask)
                
                # crop image and mask
                roi_crop_prob = self.dataset_config[sample['dataset']]['foreground_crop_prob']
                label_based_crop_prob = self.dataset_config[sample['dataset']]['label_based_crop_prob']
                uncenter_prob = self.dataset_config[sample['dataset']]['uncenter_prob']
                image, mask, y1x1z1_y2x2z2 = self._crop(image, mask, is_pos, roi_crop_prob, label_based_crop_prob, uncenter_prob)
                
                # augmentation if needed
                if sample['dataset'] in self.augmentator:
                    data_dict = {'image': image, 'label': mask}
                    aug_data_dict = self.augmentator[sample['dataset']](data_dict)
                    image, mask = aug_data_dict['image'], aug_data_dict['label']
                    
                break
            except SystemExit:
                exit()
            except:
                # record bugs in loading data
                traceback_info = traceback.format_exc()
                print(f'*** {sample["dataset"]} *** {sample["image"]} ***\n')
                print(traceback_info)

        return {'image':image, 'mask':mask, 'text':label_ls, 'modality':modality, 'image_path':sample['renorm_image'], 'mask_path':sample['renorm_segmentation_dir'], 'dataset':sample['dataset'], 'y1x1z1_y2x2z2':y1x1z1_y2x2z2}
    
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