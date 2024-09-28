#!/bin/bash
#SBATCH --job-name=eval_nano_step_140000
#SBATCH --quotatype=spot
#SBATCH --partition=medai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=30G
#SBATCH --chdir=/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/medical-universal-segmentation/log/sbatch
#SBATCH --output=/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/medical-universal-segmentation/log/sbatch/%x-%j.out
#SBATCH --error=/mnt/petrelfs/zhaoziheng/Knowledge-Enhanced-Medical-Segmentation/medical-universal-segmentation/log/sbatch/%x-%j.error
#SBATCH -x SH-IDC1-10-140-0-[132,136,137,168-177,197,199,202,222-230],SH-IDC1-10-140-1-[3,10,12,18,30,33,35,44,57,58,60,61,148,151-160,162-106,167-170,177]
###SBATCH -w SH-IDC1-10-140-0-[...], SH-IDC1-10-140-1-[...]
###SBATCH -x SH-IDC1-10-140-0-[...], SH-IDC1-10-140-1-[...]

export NCCL_DEBUG=INFO
export NCCL_IBEXT_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr  # 确保这里使用大写的MASTER_ADDR
MASTER_PORT=$((RANDOM % 101 + 20000))
export MASTER_PORT  # 导出MASTER_PORT为环境变量
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT  # 打印MASTER_PORT以便调试
echo "Rendezvous Endpoint: $MASTER_ADDR:$MASTER_PORT"


srun -p medai_llm --quotatype=spot \
--async -o /mnt/petrelfs/lihaolin/project/SAT-decompose/LoRKD-seg/log/async/inference/lorkd_imbalance_pro_inference_demo.out \
--cpus-per-task=16 \
--gres=gpu:1 torchrun \
--nnodes 1 \
--nproc_per_node 1 \
--rdzv_id 100 \
--rdzv_backend c10d \
--rdzv_endpoint=127.0.0.1:28161 inference_lora_imbalance.py \
--rcd_dir '/mnt/petrelfs/lihaolin/project/SAT-decompose/LoRKD-seg/log/inference/lorkd_imbalance_pro_inference_demo' \
--datasets_jsonl '/mnt/petrelfs/lihaolin/project/SAT-decompose/LoRKD-seg/dataset/inference_demo/demo.jsonl' \
--crop_size 288 288 96 \
--vision_backbone 'UNET_nano_lora' \
--checkpoint '/mnt/petrelfs/lihaolin/model/SAT/decompose/49_lora/lorkd-imbalance-pro.pth' \
--partial_load True \
--biolord_checkpoint '/mnt/hwfile/medai/lihaolin/models/huggingface/BioLORD-2023-C' \
--knowledge_encoder_checkpoint '/mnt/petrelfs/lihaolin/model/SAT/SAT_ckpt/pro_text_encoder.pth' \
--query_generator_partial_load True \
--region_split_json '/mnt/petrelfs/lihaolin/project/SAT-decompose/LoRKD-seg/dataset/mod_lab(49).json' \
--batchsize_3d 1 \
--max_queries 256 \
--gamma_list "4,4,4,4,4,16,12,12" \
--num_workers 8 