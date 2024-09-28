import os
import time

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from tqdm import tqdm
from einops import rearrange, repeat, reduce
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
import shutil
import pickle
from scipy.ndimage import gaussian_filter
# from thop import profile

from .metric import calculate_metric_percase
from .merge_after_evalute import merge

def compute_gaussian(tile_size, sigma_scale: float = 1. / 8, value_scaling_factor: float = 10, dtype=np.float16):
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    # gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * value_scaling_factor
    gaussian_importance_map = gaussian_importance_map.astype(dtype)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map

def convert_to_labels(dataset_name):
        #0-Abdomen, 1-Brain 2-H&N 3-LL 4-Pelvis 5-Spine 6-Thorax 7-UL
        dataset_to_label = {
            "SegRap2023_Task2": 2,
            "LNDb": 6,
            "FUMPE": 6,
            "Pancreas_CT": 0,
            "VerSe": 5,
            "SEGA": 0,
            "KiPA22": 0,
            "ATLAS": 0,
            "KiTS23": 0,
            "BraTS2023_SSA": 1,
            "BraTS2023_PED": 1,
            "BraTS2023_MEN": 1,
            "BraTS2023_GLI": 1,
            "BraTS2023_MET": 1,
            "Instance22": 1,
            "MyoPS2020": 6,
            "ATLASR2": 1,
            "LAScarQS22_Task2": 6,
            "CrossMoDA2021": 2,
            "LAScarQS22_Task1": 6,
            "MM_WHS_MRI": 6,
            "MM_WHS_CT": 6,
            "CMRxMotion": 6,
            "PARSE2022": 6,
            "ToothFairy": 2,
            "FeTA2022": 1,
            "NSCLC": 6,
            "CHAOS_CT": 0,
            "CHAOS_MRI": 0,
            "AbdomenCT1K": 0,
            "MSD_Liver": 0,
            "MSD_Lung": 6,
            "TotalSegmentator_Vertebrae": 5,
            "TotalSegmentator_Ribs": 6,
            "MSD_Colon": 0,
            "MSD_Cardiac": 6,
            "MSD_HepaticVessel": 0,
            "MSD_Pancreas": 0,
            "MSD_Spleen": 0,
            "MSD_Prostate": 4,
            "ISLES2022": 1,
            "COVID19": 6,
            "ACDC": 6,
            "MSD_Hippocampus": 1,
            "MRSpineSeg": 5,
            "PROMISE12": 4,
            "SKI10": 3,
            "SLIVER07": 0,
            "WMH_Segmentation_Challenge": 1,
            "Brain_Atlas": 1,
            "Couinaud_Liver": 0,
            # "autoPET": 8,
            #dataset with multiple region
            'BTCV_head and neck' : 2 ,
            'TotalSegmentator_Muscles_head and neck' : 2 ,
            'BrainPTM_head and neck' : 2 ,
            'AMOS22_CT_abdomen' : 0 ,
            'PDDCA_brain' : 1 ,
            'TotalSegmentator_Muscles_upper limb' : 7 ,
            'LUNA16_head and neck' : 2 ,
            'CT_ORG_abdomen' : 0 ,
            'TotalSegmentator_v2_spine' : 5 ,
            'PDDCA_head and neck' : 2 ,
            'SegTHOR_thorax' : 6 ,
            'TotalSegmentator_Organs_pelvis' : 4 ,
            'BTCV_Cervix_abdomen' : 0 ,
            'TotalSegmentator_Cardiac_thorax' : 6 ,
            'SegTHOR_abdomen' : 0 ,
            'DAP_Atlas_lower limb' : 3 ,
            'SegRap2023_Task1_spine' : 5 ,
            'DAP_Atlas_thorax' : 6 ,
            'TotalSegmentator_v2_head and neck' : 2 ,
            'Hecktor2022_head and neck' : 2 ,
            'DAP_Atlas_upper limb' : 7 ,
            'HAN_Seg_head and neck' : 2 ,
            'TotalSegmentator_v2_thorax' : 6 ,
            'SegRap2023_Task1_brain' : 1 ,
            'SegTHOR_head and neck' : 2 ,
            'WORD_lower limb' : 3 ,
            'BrainPTM_brain' : 1 ,
            'DAP_Atlas_head and neck' : 2 ,
            'AMOS22_CT_head and neck' : 2 ,
            'TotalSegmentator_Organs_head and neck' : 2 ,
            'TotalSegmentator_Muscles_pelvis' : 4 ,
            'TotalSegmentator_Cardiac_pelvis' : 4 ,
            'LUNA16_thorax' : 6 ,
            'BTCV_abdomen' : 0 ,
            'AMOS22_CT_pelvis' : 4 ,
            'WORD_abdomen' : 0 ,
            'TotalSegmentator_v2_pelvis' : 4 ,
            'WORD_head and neck' : 2 ,
            'TotalSegmentator_Muscles_thorax' : 6 ,
            'DAP_Atlas_abdomen' : 0 ,
            'TotalSegmentator_Organs_abdomen' : 0 ,
            'TotalSegmentator_Muscles_lower limb' : 3 ,
            'AMOS22_MRI_pelvis' : 4 ,
            'TotalSegmentator_Cardiac_abdomen' : 0 ,
            'CT_ORG_pelvis' : 4 ,
            'CTPelvic1K_spine' : 5 ,
            'TotalSegmentator_v2_abdomen' : 0 ,
            'SegRap2023_Task1_head and neck' : 2 ,
            'WORD_pelvis' : 4 ,
            'HAN_Seg_brain' : 1 ,
            'BTCV_Cervix_pelvis' : 4 ,
            'CT_ORG_thorax' : 6 ,
            'FLARE22_abdomen' : 0 ,
            'CT_ORG_head and neck' : 2 ,
            'AMOS22_MRI_thorax' : 6 ,
            'TotalSegmentator_Organs_thorax' : 6 ,
            'HAN_Seg_spine' : 5 ,
            'DAP_Atlas_pelvis' : 4 ,
            'AMOS22_MRI_abdomen' : 0 ,
            'CTPelvic1K_pelvis' : 4 ,
            'FLARE22_head and neck' : 2 ,
            'DAP_Atlas_spine' : 5 ,
        }
        task_label = dataset_to_label[dataset_name]
        task_label_one_hot = F.one_hot(torch.tensor(task_label), num_classes = 8)
        return task_label_one_hot

def inference(model, 
             query_generator, 
             device, 
             testset, 
             testloader, 
             nib_dir):
    
    if int(os.environ["LOCAL_RANK"]) == 0:
        jsonl_file = testset.jsonl_file.split('/')[-1]
        shutil.copy(testset.jsonl_file, f'{nib_dir}/{jsonl_file}')
        
    # datasets -> modality
    datasets2modality = {}
    
    model.eval()
    query_generator.eval()
        
    with torch.no_grad():
        data_time = 0
        pred_time = 0
        metric_time = 0
        
        avg_patch_batch_num = 0
        avg_query_batch_num = 0
        
        # in ddp, only master process display the progress bar
        if int(os.environ["RANK"]) == 0:
            testloader = tqdm(testloader, disable=False)
        else:
            testloader = tqdm(testloader, disable=True)  
    
        # gaussian kernel to accumulate predcition
        gaussian = torch.tensor(compute_gaussian((288, 288, 96))).to(device)    # hwd

        end_time = time.time()
        for sample in testloader:    # in evaluation/inference, a "batch" in loader is a volume
            # data loading
            dataset_name = sample['dataset_name']
            task_label = convert_to_labels(dataset_name)
            sample_id = sample['sample_id'] 
            batched_patches = sample['batched_patches']
            batched_y1y2_x1x2_z1z2 = sample['batched_y1y2_x1x2_z1z2']
            split_labels = sample['split_labels'] 
            split_n1n2 = sample['split_n1n2']
            labels = sample['labels']
            modality = sample['modality']
            
            if dataset_name not in datasets2modality:
                datasets2modality[dataset_name] = modality

            _, h, w, d = sample['chwd']
            n = len(labels)
            
            prediction = torch.zeros((n, h, w, d))
            accumulation = torch.zeros((n, h, w, d))
            
            # N,H,W,D = gt_segmentation.shape
            # prediction = torch.zeros((N,H,W,D))
            # accumulation = torch.zeros((N,H,W,D))
            
            data_time += (time.time()-end_time) 
            end_time = time.time()
            
            avg_patch_batch_num += len(batched_patches)
            avg_query_batch_num += len(split_labels)
            
            with autocast():
                
                # for each batch of queries
                queries_ls = []
                for labels_ls, n1n2 in zip(split_labels, split_n1n2):  # convert list of texts to list of embeds
                    queries_ls.append(query_generator.get_query(labels_ls, [modality]))

                torch.cuda.empty_cache()
                # for each batch of patches, query with all labels
                for patches, y1y2_x1x2_z1z2_ls in zip(batched_patches, batched_y1y2_x1x2_z1z2):   # [b, c, h, w, d]
                    patches = patches.to(device=device)
                    b, c, _, _, _ = patches.shape
                    task_labels = task_label.repeat(b, 1)
                    # print('task_labels.shape=', task_labels.shape)
                    prediction_patch = model(queries=queries_ls, image_input=patches, task_labels=task_labels)
                    prediction_patch = torch.sigmoid(prediction_patch)  # bnhwd
                    prediction_patch = prediction_patch.detach() # .cpu().numpy()
                    
                    # fill in 
                    for b in range(len(y1y2_x1x2_z1z2_ls)):
                        y1, y2, x1, x2, z1, z2 = y1y2_x1x2_z1z2_ls[b]

                        # gaussian accumulation
                        tmp = prediction_patch[b, :, :y2-y1, :x2-x1, :z2-z1] * gaussian[:y2-y1, :x2-x1, :z2-z1] # accelerated by gpu
                        prediction[:, y1:y2, x1:x2, z1:z2] += tmp.cpu()
                        accumulation[:, y1:y2, x1:x2, z1:z2] += gaussian[:y2-y1, :x2-x1, :z2-z1].cpu()

                        # prediction[n1:n2, y1:y2, x1:x2, z1:z2] += prediction_patch[b, :n2-n1, :y2-y1, :x2-x1, :z2-z1]
                        # accumulation[n1:n2, y1:y2, x1:x2, z1:z2] += 1
            
            pred_time += (time.time()-end_time)
            end_time = time.time()
                            
            # avg            
            prediction = prediction / accumulation
            prediction = torch.where(prediction>0.5, 1.0, 0.0)
            prediction = prediction.numpy()
            
            # # cal metrics : [{'dice':x, ...}, ...] 
            # scores = []
            # for j in range(len(labels)):
            #     scores.append(calculate_metric_percase(prediction[j, :, :, :], gt_segmentation[j, :, :, :], dice_score, nsd_score))    # {'dice':0.9, 'nsd':0.8} 每个label一个dict
            
            # visualization  
            Path(f'{nib_dir}/{dataset_name}').mkdir(exist_ok=True, parents=True)
            results = np.zeros((h, w, d)) # hwd
            for j, label in enumerate(labels):
                results += prediction[j, :, :, :] * (j+1)   # 0 --> 1 (skip background)

                    # prediction_cropped = prediction[j, :288, :288, :96]
                    # results += prediction_cropped * (j + 1)

                Path(f'{nib_dir}/{dataset_name}/seg_{sample_id}').mkdir(exist_ok=True, parents=True)
                # 每个label单独一个nii.gz
                segobj = nib.nifti2.Nifti1Image(prediction[j, :, :, :], np.eye(4))
                nib.save(segobj, f'{nib_dir}/{dataset_name}/seg_{sample_id}/{label}.nii.gz')
            segobj = nib.nifti2.Nifti1Image(results, np.eye(4))
            nib.save(segobj, f'{nib_dir}/{dataset_name}/seg_{sample_id}.nii.gz')
                
            # image = testset.load_image(image_path)
            # image = np.squeeze(image)
            image = sample['image'].numpy()
            if image.ndim == 4:
                image = image[0, :, :, :]   # h w d
            imgobj = nib.nifti2.Nifti1Image(image, np.eye(4))
            nib.save(imgobj, f'{nib_dir}/{dataset_name}/img_{sample_id}.nii.gz')
            
        torch.cuda.empty_cache()   