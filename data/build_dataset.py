import os

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .dataset_3d import Med_SAM_Dataset
from .dataset_downstream import Med_SAM_Dataset_downstream
from .dataset_3d_region import Med_SAM_Dataset_region
from .dataset_3d_noshuffle import Med_SAM_Dataset_noshuffle
from .sampler import BatchSampler, BatchSampler_noshuffle
from .collect_3d import collect_fn

def build_dataset_noshuffle(args):
    dataset = Med_SAM_Dataset_noshuffle(jsonl_file=args.datasets_jsonl, 
                              crop_size=args.crop_size,    # h w d
                              max_queries=args.max_queries,
                              dataset_config=args.dataset_config,
                              allow_repeat=args.allow_repeat
                              )
    if "RANK" not in os.environ:
        # print('Using BatchSampler')
        data_split_in_an_epoch = dataset.data_split
        batchsize = {'3D':args.batchsize_3d, '2D':args.batchsize_2d}
        sampler = BatchSampler_noshuffle(data_split_in_an_epoch, batchsize, True)
        if args.num_workers is not None:
            dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collect_fn, pin_memory=args.pin_memory, num_workers=args.num_workers)
        else:
            dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collect_fn, pin_memory=args.pin_memory)
    else:
        # print('Using DistributedSampler')
        sampler = DistributedSampler(dataset, shuffle=False)
        if args.num_workers is not None:
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batchsize_3d, collate_fn=collect_fn, pin_memory=args.pin_memory, num_workers=args.num_workers)
        else:
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batchsize_3d, collate_fn=collect_fn, pin_memory=args.pin_memory)
    
    return dataset, dataloader, sampler


def build_dataset(args):
    dataset = Med_SAM_Dataset(jsonl_file=args.datasets_jsonl, 
                              crop_size=args.crop_size,    # h w d
                              max_queries=args.max_queries,
                              dataset_config=args.dataset_config,
                              allow_repeat=args.allow_repeat
                              )
    if "RANK" not in os.environ:
        data_split_in_an_epoch = dataset.data_split
        batchsize = {'3D':args.batchsize_3d, '2D':args.batchsize_2d}
        sampler = BatchSampler(data_split_in_an_epoch, batchsize, True)
        if args.num_workers is not None:
            dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collect_fn, pin_memory=args.pin_memory, num_workers=args.num_workers)
        else:
            dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collect_fn, pin_memory=args.pin_memory)
    else:
        sampler = DistributedSampler(dataset)
        if args.num_workers is not None:
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batchsize_3d, collate_fn=collect_fn, pin_memory=args.pin_memory, num_workers=args.num_workers)
        else:
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batchsize_3d, collate_fn=collect_fn, pin_memory=args.pin_memory)
    
    return dataset, dataloader, sampler

def build_dataset_downstream(args):
    dataset = Med_SAM_Dataset_downstream(jsonl_file=args.datasets_jsonl, 
                              crop_size=args.crop_size,    # h w d
                              max_queries=args.max_queries,
                              dataset_config=args.dataset_config,
                              allow_repeat=args.allow_repeat
                              )
    if "RANK" not in os.environ:
        data_split_in_an_epoch = dataset.data_split
        batchsize = {'3D':args.batchsize_3d, '2D':args.batchsize_2d}
        sampler = BatchSampler(data_split_in_an_epoch, batchsize, True)
        if args.num_workers is not None:
            dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collect_fn, pin_memory=args.pin_memory, num_workers=args.num_workers)
        else:
            dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collect_fn, pin_memory=args.pin_memory)
    else:
        sampler = DistributedSampler(dataset)
        if args.num_workers is not None:
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batchsize_3d, collate_fn=collect_fn, pin_memory=args.pin_memory, num_workers=args.num_workers)
        else:
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batchsize_3d, collate_fn=collect_fn, pin_memory=args.pin_memory)
    
    return dataset, dataloader, sampler

def build_dataset_region(args):
    dataset = Med_SAM_Dataset_region(jsonl_file=args.datasets_jsonl, 
                              crop_size=args.crop_size,    # h w d
                              max_queries=args.max_queries,
                              dataset_config=args.dataset_config,
                              allow_repeat=args.allow_repeat
                              )
    if "RANK" not in os.environ:
        data_split_in_an_epoch = dataset.data_split
        batchsize = {'3D':args.batchsize_3d, '2D':args.batchsize_2d}
        sampler = BatchSampler(data_split_in_an_epoch, batchsize, True)
        if args.num_workers is not None:
            dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collect_fn, pin_memory=args.pin_memory, num_workers=args.num_workers)
        else:
            dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collect_fn, pin_memory=args.pin_memory)
    else:
        sampler = DistributedSampler(dataset)
        if args.num_workers is not None:
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batchsize_3d, collate_fn=collect_fn, pin_memory=args.pin_memory, num_workers=args.num_workers)
        else:
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batchsize_3d, collate_fn=collect_fn, pin_memory=args.pin_memory)
    
    return dataset, dataloader, sampler

def build_dataset_order(args, region):
    if region == 0:
        region_dataset = args.datasets_jsonl + '/ablation_49_split-abd.jsonl'
        print("Using region Abdomen!")
    elif region == 1:
        region_dataset = args.datasets_jsonl + '/ablation_49_split-brain.jsonl'
        print("Switching to region brain!")
    elif region == 2:
        region_dataset = args.datasets_jsonl + '/ablation_49_split-hn.jsonl'
        print("Switching to region head&neck!")
    elif region == 3:
        region_dataset = args.datasets_jsonl + '/ablation_49_split-ll.jsonl'
        print("Switching to region lower limb!")
    elif region == 4:
        region_dataset = args.datasets_jsonl + '/ablation_49_split-pelvis.jsonl'
        print("Switching to region pelvis!")
    elif region == 5:
        region_dataset = args.datasets_jsonl + '/ablation_49_split-spine.jsonl'
        print("Switching to region spine!")
    elif region == 6:
        region_dataset = args.datasets_jsonl + '/ablation_49_split-thorax.jsonl'
        print("Switching to region thorax!")
    elif region == 7:
        region_dataset = args.datasets_jsonl + '/ablation_49_split-ul.jsonl'
        print("Switching to region upper limb!")
    elif region == 8:
        region_dataset = args.datasets_jsonl + '/ablation_49_split-abd.jsonl'
        print("Training done ! Switching to first region Abdomen!")
    else:
        # print('No specific region!!!')
        raise ValueError("No specific region!!!")
    dataset = Med_SAM_Dataset(jsonl_file=region_dataset, 
                              crop_size=args.crop_size,    # h w d
                              max_queries=args.max_queries,
                              dataset_config=args.dataset_config,
                              allow_repeat=args.allow_repeat
                              )
    if "RANK" not in os.environ:
        data_split_in_an_epoch = dataset.data_split
        batchsize = {'3D':args.batchsize_3d, '2D':args.batchsize_2d}
        sampler = BatchSampler(data_split_in_an_epoch, batchsize, True)
        if args.num_workers is not None:
            dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collect_fn, pin_memory=args.pin_memory, num_workers=args.num_workers)
        else:
            dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collect_fn, pin_memory=args.pin_memory)
    else:
        sampler = DistributedSampler(dataset)
        if args.num_workers is not None:
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batchsize_3d, collate_fn=collect_fn, pin_memory=args.pin_memory, num_workers=args.num_workers)
        else:
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batchsize_3d, collate_fn=collect_fn, pin_memory=args.pin_memory)
    
    return dataset, dataloader, sampler