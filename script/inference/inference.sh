#!/bin/bash

set -e -x

export PYTHONPATH=$PWD:$PYTHONPATH
export WANDB_MODE="disabled"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FINETRAINERS_LOG_LEVEL="DEBUG"

BACKEND="ptd"
NUM_GPUS=1
CUDA_VISIBLE_DEVICES="0"

# Dataset paths
DATASET_FILE="/path/to/inference.json"

DDP_1="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 1 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_2="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 2 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_4="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 4 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_8="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 8 --dp_shards 1 --cp_degree 1 --tp_degree 1"

# Parallel arguments
parallel_cmd=(
  $DDP_1
)

# Model arguments
model_cmd=(
  --model_name "wan"
  --pretrained_model_name_or_path "./ckpt/wan2.1-t2v-1.3b-diffusers"
  --enable_slicing
  --enable_tiling
  --dino_in_channels 8
  --dino_out_channels 8
  --vggt_in_channels 8
  --vggt_out_channels 8
  --flow_in_channels 16
  --flow_out_channels 16
)

# Inference arguments
inference_cmd=(
  --inference_type text_to_video
  --dataset_file "$DATASET_FILE"
)

# Attention provider arguments
attn_provider_cmd=(
  --attn_provider native
)

# Torch config arguments
torch_config_cmd=(
  --allow_tf32
  --float32_matmul_precision high
)

# Miscellaneous arguments 包含 lora_path
miscellaneous_cmd=(
  --seed 42
  --tracker_name "dreamworld-inference"
  --lora_path "/path/to/pytorch_lora_weights.safetensors"
  --output_dir "/path/to/output"
  --init_timeout 600
  --nccl_timeout 600
  --report_to "wandb"
)

# Execute the inference script
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=$NUM_GPUS \
  --rdzv_backend c10d \
  --rdzv_endpoint="localhost:19242" \
  ./script/inference/inference.py \
    "${parallel_cmd[@]}" \
    "${model_cmd[@]}" \
    "${inference_cmd[@]}" \
    "${attn_provider_cmd[@]}" \
    "${torch_config_cmd[@]}" \
    "${miscellaneous_cmd[@]}"

echo -ne "-------------------- Finished executing script --------------------\n\n"