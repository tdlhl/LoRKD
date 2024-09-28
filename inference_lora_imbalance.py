import os
import datetime
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path

from data.dataset_for_ddp_evaluate_3d import Splited_Images_and_Labels_3D, collate_splited_images_and_labels
from data.inference_dataset import Inference_Dataset, collate_fn

from model.build_model import build_maskformer, load_text_encoder
from model.query_generator import Query_Generator

from evaluate.inference_lora import inference
from evaluate.params import parse_args

from train.dist import is_ddp, is_master
from peft.lora_fast import LoraConv3d, MultiLoraConv3d_imbalance
from peft.unet_adapter import AdapterWrapperUNet_imbalance


def set_seed(config):
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    # new seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def main(args):
    # set gpu
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device=torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    gpu_id = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group(backend="nccl", init_method='env://', timeout=datetime.timedelta(seconds=7200))   # might takes a long time to sync between process
    # dispaly
    if int(os.environ["LOCAL_RANK"]) == 0:
        print('** GPU NUM ** : ', torch.cuda.device_count())  
        print('** WORLD SIZE ** : ', torch.distributed.get_world_size())
    rank = int(os.environ["LOCAL_RANK"])
    print(f"** DDP ** : Start running DDP on rank {rank}.")
    
    if int(os.environ["LOCAL_RANK"]) == 0:
        Path(args.rcd_dir).mkdir(exist_ok=True, parents=True)
    
    # dataset and loader
    testset = Inference_Dataset(args.datasets_jsonl, args.max_queries, args.batchsize_3d)
    sampler = DistributedSampler(testset)
    testloader = DataLoader(testset, sampler=sampler, batch_size=1, pin_memory=args.pin_memory, num_workers=args.num_workers, collate_fn=collate_splited_images_and_labels, shuffle=False)
    sampler.set_epoch(0)
    
    model = build_maskformer(args, device, gpu_id)
    adapter_class = MultiLoraConv3d_imbalance
    model_lora = AdapterWrapperUNet_imbalance(model, adapter_class, num_tasks=args.num_tasks, gamma=args.gamma_list, lora_alpha=args.lora_alpha)

    total_num = sum(p.numel() for p in model_lora.parameters())
    print(f"* Decomposed model (add all LoRA) ** {total_num/1e6}M parameters")
    
    # load knowledge encoder
    query_generator = Query_Generator(cpt_checkpoint=args.cpt_checkpoint,
                                      finetuned_cpt_checkpoint=args.finetuned_cpt_checkpoint,
                                      knowledge_encoder_checkpoint=args.knowledge_encoder_checkpoint,
                                      pubmedbert_checkpoint=args.pubmedbert_checkpoint,
                                      biolord_checkpoint=args.biolord_checkpoint,
                                      basebert_checkpoint=args.basebert_checkpoint,
                                      finetuned_basebert_checkpoint=args.finetuned_basebert_checkpoint,
                                      random_embed_label_mapping=args.random_embed_label_mapping,
                                      finetuned_random_embed=args.finetuned_random_embed,
                                      partial_load=args.query_generator_partial_load,
                                      gpu_id=gpu_id,
                                      device=device)
    
    # load checkpoint if specified
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if args.partial_load:
        model_dict =  model_lora.state_dict()
        # check difference
        unexpected_state_dict = [k for k in checkpoint['model_state_dict'].keys() if k not in model_dict.keys()]
        print('The following parameters are unexpected in checkpoint:\n', unexpected_state_dict)
        missing_state_dict = [k for k in model_dict.keys() if k not in checkpoint['model_state_dict'].keys()]
        print('The following parameters are missing in checkpoint:\n', missing_state_dict)
        unmatchd_state_dict = [k for k,v in checkpoint['model_state_dict'].items() if k in model_dict.keys() and v.shape != model_dict[k].shape]
        print('The following parameters have different shapes in checkpoint:\n', unmatchd_state_dict)
        # load partial parameters
        state_dict = {k:v for k,v in checkpoint['model_state_dict'].items() if k in model_dict.keys() and v.shape == model_dict[k].shape}
        print('The following parameters are loaded :\n', state_dict.keys())
        model_dict.update(state_dict)
        model_lora.load_state_dict(model_dict)
    else:
        model_lora.load_state_dict(checkpoint['model_state_dict'])
    
    # choose how to evaluate the checkpoint
    inference(model=model_lora,
             query_generator=query_generator,
             device=device,
             testset=testset,
             testloader=testloader,
             nib_dir=args.rcd_dir)

if __name__ == '__main__':
    # Some Important Args
    # --checkpoint
    # --datasets_jsonl
    # --max_queries 24
    # --batchsize_3d 1
    # --batchsize_2d 1
    # --rcd_dir
    # --rcd_file
    
    # get configs
    args = parse_args()
    
    main(args)

    
    
    
        
    
    