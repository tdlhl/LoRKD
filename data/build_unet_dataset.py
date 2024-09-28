import os

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .dataset_3d_unet import Med_SAM_Dataset
from .sampler import BatchSampler
from .collect_3d_unet import collect_fn


def build_unet_dataset(args):
    dataset = Med_SAM_Dataset(jsonl_file=args.datasets_jsonl, 
                              label_json=args.label_json,
                              crop_size=args.crop_size,    # h w d
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