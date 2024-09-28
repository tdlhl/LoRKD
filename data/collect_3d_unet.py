import torch
import torch.nn.functional as F
from einops import repeat

def collect_fn(data):
    """
    Pad images and masks to the same depth and num of class
    
    Args:
        data : 'image':image, 'mask':mask, 'label':sample['label'], 'image_path':sample['renorm_image'], 'mask_path':sample['renorm_segmentation_dir'], 'dataset':sample['dataset'], ['c1_c2']:[c1, c2], ['y1x1z1_y2x2z2']:y1x1z1_y2x2z2
    """
    
    image = []
    gt_ls = []
    label_ls = []
    image_path_ls = []
    mask_path_ls = []
    dataset_ls = []
    c1_c2_ls = []
    y1x1z1_y2x2z2_ls = []

    for i, sample in enumerate(data):
        if sample['image'].shape[0] == 1:
            sample['image'] = repeat(sample['image'], 'c h w d -> (c r) h w d', r=3)
        image.append(sample['image'])
        gt_ls.append(sample['mask'].float())
        
        label_ls.append(sample['label']) 
        image_path_ls.append(sample['image_path'])
        mask_path_ls.append(sample['mask_path'])
        dataset_ls.append(sample['dataset'])
        c1_c2_ls.append(sample['c1_c2'])
        y1x1z1_y2x2z2_ls.append(sample['y1x1z1_y2x2z2'])
    
    image = torch.stack(image, dim=0)
    return {
        'image':image, 
        'gt_ls':gt_ls, 
        'label_ls':label_ls, 
        'image_path':image_path_ls, 
        'mask_path':mask_path_ls, 
        'dataset':dataset_ls, 
        'y1x1z1_y2x2z2':y1x1z1_y2x2z2_ls,
        'c1_c2':c1_c2_ls
        }
