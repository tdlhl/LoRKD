'''
Parameters assignment
'''

import argparse

def str2bool(v):
    return v.lower() in ('true')

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Exp Controller
    
    parser.add_argument(
        "--rcd_dir",
        type=str,
        default=None,
        help="save the evaluation results (in a directory)",
    )
    parser.add_argument(
        "--rcd_file",
        type=str,
        default=None,
        help="save the evaluation results (in a csv/xlsx file)",
    )
    parser.add_argument(
        "--visualization",
        type=str2bool,
        default=False,
        help="save the visualization for each case (img, gt, pred)",
    )
    parser.add_argument(
        "--resume",
        type=str2bool,
        default=False,
        help="resume",
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
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
    )
    
    # Metrics
    
    parser.add_argument(
        "--dice",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--nsd",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--region_split_json",
        type=str,
        default='/mnt/petrelfs/lihaolin/project/SAT-decompose/LoRKD-seg/dataset/mod_lab(49).json',
    )
    
    # Med SAM Dataset
    
    parser.add_argument(
        "--datasets_jsonl",
        type=str,
        default='/mnt/petrelfs/lihaolin/project/SAT-decompose/LoRKD-seg/dataset/inference_demo/demo.jsonl',
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
        default=32,
    )
    parser.add_argument(
        "--batchsize_3d",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--pin_memory",
        type=str2bool,
        default=False,
        help='load data to gpu to accelerate'
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default='/remote-home/zihengzhao/Knowledge-Enhanced-Medical-Segmentation/medical-knowledge-pretraining/src/others/pubmed_bert',
    )
    parser.add_argument(
        "--teacher_text_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--query_generator_partial_load",
        type=str2bool,
        default=False,
        help="Allow to load partial paramters from checkpoint",
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
        "--knowledge_encoder_checkpoint",
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

    # MaskFormer
    
    parser.add_argument(
        "--vision_backbone",
        type=str,
        help='UNET or UNET-H'
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

    #decompose&lora params
    parser.add_argument(
        "--lora_on",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--backbone_checkpoint",
        type=str,
        default=None,
        help="Checkpoint path",
    )
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
        "--gamma_list", 
        type=str, 
        default="8,8,8,8,8,8,8,8",  
        help="low rank dimension r[]"
    )
    parser.add_argument(
        "--lora_alpha", 
        type=int, 
        default=16, 
        help="scaling factor"
    )

    #grid_search
    parser.add_argument(
        "--label_statistic_json",
        type=str,
        default='/mnt/hwfile/medai/zhaoziheng/SAM/processed_files_v4/mod_lab_accum_statis(72).json',
    )
    
    args = parser.parse_args()
    args.gamma_list = [int(x) for x in args.gamma_list.split(',')]
    return args