# export
export $(grep -v '^#' .env | xargs)

# generate videos
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python world_generators/generate_videos.py \
  --model-name wan2.1_t2v \
  --pretrained_path="/path/to/Wan2.1-T2V-1.3B-Diffusers" \
  --lora_path="/path/to/your/pytorch_lora_weights.safetensors" \

#evaluate
python worldscore/run_evaluate.py --model_name wan2.1_t2v