import torch
import torch.nn as nn
import time
import os
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np

from .maskformer import Maskformer
from .text_tower import Text_Tower
from .SAT import SAT

from train.dist import is_master, is_ddp


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def build_teacher(args, device, gpu_id):
    model = Maskformer(args.teacher_backbone, args.crop_size, args.patch_size, args.learnable_pe, args.deep_supervision)
    for param in model.parameters():
        param.requires_grad = False  
    device = torch.device('cuda')
    model.to(device)
        
    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    
    if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
        print(f"** Teacher ** {get_parameter_number(model)['Total']/1e6}M parameters ** Trainable {get_parameter_number(model)['Trainable']/1e6}M parameters")
            
    return model

def build_student(args, device, gpu_id):
    model = Maskformer(args.student_backbone, args.crop_size, args.patch_size, args.learnable_pe, args.deep_supervision)
    
    if "RANK" in os.environ:
        model = model.to(device)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)        
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
        print('using torch.nn.parallel.DistributedDataParallel! on ', gpu_id)
    else:
        device = torch.device('cuda')
        model = nn.DataParallel(model)
        model.to(device)
        print('using device:',device)
        
    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    
    if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
        print(f"** Student(before add LoRA) ** {get_parameter_number(model)['Total']/1e6}M parameters ** Trainable {get_parameter_number(model)['Trainable']/1e6}M parameters")
            
    return model

def build_maskformer(args, device, gpu_id):
    model = Maskformer(args.vision_backbone, args.crop_size, args.patch_size, args.learnable_pe, args.deep_supervision)
    
    if "RANK" in os.environ:
        model = model.to(device)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)        
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
        print('build_maskformer device_ids=', gpu_id)
    else:
        device = torch.device('cuda')
        model = nn.DataParallel(model)
        model.to(device)
        
    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    
    if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
        print(f"** MODEL ** {get_parameter_number(model)['Total']/1e6}M parameters ** Trainable {get_parameter_number(model)['Trainable']/1e6}M parameters")
            
    return model

def load_text_encoder(args, device, gpu_id):
    model = Text_Tower(args.tokenizer_name, 768)
    
    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)        
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
    print('load_text_encoder device_ids=', gpu_id)
    checkpoint = torch.load(args.teacher_text_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if int(os.environ["RANK"]) == 0:
        print(f"** Model ** Load pretrained text encoder from {args.teacher_text_checkpoint}.")
        
    return model

def build_segmentation_model(args, device, gpu_id):
    model = SAT(args.vision_backbone, args.crop_size)
    
    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)        
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
            
    return model

def load_checkpoint(checkpoint, 
                    resume, 
                    partial_load, 
                    model, 
                    device,
                    optimizer=None,
                    ):
    
    if is_master():
        print('** CHECKPOINT ** : Load checkpoint from %s' % (checkpoint))
    
    checkpoint = torch.load(checkpoint, map_location=device)
        
    # load part of the checkpoint
    if partial_load:
        model_dict =  model.state_dict()
        # check difference
        unexpected_state_dict = [k for k in checkpoint['model_state_dict'].keys() if k not in model_dict.keys()]
        missing_state_dict = [k for k in model_dict.keys() if k not in checkpoint['model_state_dict'].keys()]
        unmatchd_state_dict = [k for k,v in checkpoint['model_state_dict'].items() if k in model_dict.keys() and v.shape != model_dict[k].shape]
        # load partial parameters
        state_dict = {k:v for k,v in checkpoint['model_state_dict'].items() if k in model_dict.keys() and v.shape == model_dict[k].shape}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        if is_master():
            print('The following parameters are unexpected in SAT checkpoint:\n', unexpected_state_dict)
            print('The following parameters are missing in SAT checkpoint:\n', missing_state_dict)
            print('The following parameters have different shapes in SAT checkpoint:\n', unmatchd_state_dict)
            print('The following parameters are loaded in SAT:\n', state_dict.keys())
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        
    # if resume, load optimizer and step
    if resume:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = int(checkpoint['step']) + 1
    else:
        start_step = 1
        
    return model, optimizer, start_step