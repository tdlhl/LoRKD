'''
Parameters assignment
'''

import argparse
import os

def str2bool(v):
    return v.lower() in ('true')

"""
def get_default_params(model_name):
    # Params from paper [CLIP](https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}
"""

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Exp ID
    
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of the exp",
    )
    
    # Checkpoint
    
    parser.add_argument(
        "--resume",
        type=str2bool,
        default=False,
        help="Resume an interrupted exp",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path",
    )
    parser.add_argument(
        "--partial_load",
        type=str2bool,
        default=False,
        help="Allow to load partial paramters from checkpoint",
    )
    
    # Save and Log
    
    parser.add_argument(
        "--save_large_interval", 
        type=int, 
        default=1000, 
        help="Save checkpoint regularly"
    )
    parser.add_argument(
        "--save_small_interval", 
        type=int, 
        default=None, 
        help="Save checkpoint more frequentlt in case of interruption (not necessary)"
    )
    parser.add_argument(
        "--log_step_interval", 
        type=int, 
        default=1000, 
        help="Output and record log info every N steps"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default='log',
        help="Dir of exp logs",
    )
    
    # Randomness
    
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed",
    )
    
    # GPU
    
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
    )
    
    # Loss function, Optimizer
    
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1.0e-8
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--accumulate_grad_interval", 
        type=int, 
        default=1, 
        help="accumulate grad"
    )
    
    # Learing Rate 
    
    parser.add_argument(
        "--step_num",
        type=int,
        default=100000,
        help="Total step num",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5000,
        help="Warm up step num",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--lr_ratio",
        type=float,
        default=8,
        help="Peak learning rate",
    )
    
    # Med SAM Dataset
    
    parser.add_argument(
        "--datasets_jsonl",
        type=str,
        default='/mnt/petrelfs/lihaolin/project/SAT-decompose/LoRKD-seg/dataset/inference_demo/demo.jsonl',
    )
    parser.add_argument(
        "--label_json",
        type=str,
        default='/mnt/petrelfs/lihaolin/project/SAT-decompose/LoRKD-seg/dataset/mod_lab(49).json',
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help='path to dataset config parameters'
    )
    
    # Sampler and Loader
    
    parser.add_argument(
        "--crop_size",
        type=int,
        nargs='+',
        default=[288, 288, 96],
        help='h and w will be normalized to the fixed size, d will be truncated by a max depth',
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--batchsize_3d",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--batchsize_2d",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--pin_memory",
        type=str2bool,
        default=True,
        help='load data to gpu to accelerate'
    )
    parser.add_argument(
        "--allow_repeat",
        type=str2bool,
        default=True,
        help='repeat multipy times for sample with too many labels (to accelerate convergency)'
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None
    )
    
    # Knowledge Encoder (if given checkpoint) or Load Embeddings
    """
    1. 使用fixed text encoder --> knowledge_embeddings_dir
    2. 使用open text encoder(initialized or resume training) --> knowledge_encoder_checkpoint
    3. 使用fixed cpt + modality embed --> cpt_embeddings_dir + finetuned_cpt_checkpoint(if resume training)
    4. 使用open cpt + modality embed --> cpt_checkpoint + finetuned_cpt_checkpoint(if resume training) + partial_load(if inheriting modality embeddings)
    5. 使用fixed basebert + modality embed --> basebert_embeddings_dir + finetuned_basebert_checkpoint(if resume training)
    6. 使用open basebert + modality embed --> basebert_checkpoint + finetuned_basebert_checkpoint(if resume training) + partial_load(if inheriting modality embeddings)
    7. 使用random initialized and learnable embed --> random_embed_label_mapping + finetuned_random_embed(if resume training)
    """
    
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default='/mnt/petrelfs/lihaolin/model/transformer_cache/pubmed_bert',
    )
    parser.add_argument(
        "--max_text_length",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--cpt_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--cpt_embeddings_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--finetuned_cpt_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--knowledge_embeddings_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--biolord_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pubmedbert_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--knowledge_encoder_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--teacher_nano_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--teacher_text_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--basebert_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--basebert_embeddings_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--finetuned_basebert_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--random_embed_label_mapping",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--finetuned_random_embed",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--open_bert_layer",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--open_modality_embed",
        type=str2bool,
        default=False,
        help="open modality embed in knowledge encoder",
    )
    parser.add_argument(
        "--query_generator_partial_load",
        type=str2bool,
        default=False,
        help="Allow to load partial paramters (only train modality embed -> train cpt)",
    )
    parser.add_argument(
        "--pretrained_visual_encoder",  # 沿用VLP中的UNET Encoder
        type=str2bool,
        default=False,
    )
    
    # MaskFormer
    
    parser.add_argument(
        "--teacher_backbone",
        type=str,
        default='UNET',
        help='UNET or SwinUNETR'
    )
    parser.add_argument(
        "--student_backbone",
        type=str,
        default='UNET_lora',
        help='UNET or SwinUNETR'
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs='+',
        default=[32, 32, 32],
        help='patch size on h w and d'
    )
    parser.add_argument(
        "--learnable_pe",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--deep_supervision",
        type=str2bool,
        default=False,
    )

    # KD params
    parser.add_argument(
        "--kd_type",
        type=str,
        default='KD',
        help='KD, SKD, CWD, Fit'
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature for KD",
    )
    parser.add_argument(
        "--lambda_kd",
        type=float,
        default=0.1,
        help="Weight for KD loss",
    )

    #decompose&lora params
    parser.add_argument(
        "--num_tasks", 
        type=int, 
        default=8, 
        help="task number(lora number)"
    )
    parser.add_argument(
        "--gamma", 
        type=int, 
        default=8, 
        help="low rank dimension r"
    )
    parser.add_argument(
        "--lora_alpha", 
        type=int, 
        default=16, 
        help="scaling factor"
    )
    parser.add_argument(
        "--gamma_list", 
        type=str,  # 接收字符串
        default="8,8,8,8,8,8,8,8",  # 使用字符串形式的默认值
        help="low rank dimension r[]"
    )
    parser.add_argument(
        "--region_step",
        type=int,
        nargs='+',
        default=None,
        help='patch size on h w and d'
    )
    parser.add_argument(
        "--resume_region", 
        type=int, 
        default=0, 
        help="resume_region"
    )

    parser.add_argument(
        "--region_split_json",
        type=str,
        default='/mnt/petrelfs/lihaolin/project/SAT-decompose/LoRKD-seg/dataset/mod_lab(49).json',
    )
    
    args = parser.parse_args()
    args.gamma_list = [int(x) for x in args.gamma_list.split(',')]
    return args