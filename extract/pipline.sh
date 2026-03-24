#DINOv2
cd DINOv2
python pca.py \
    --video_folder "../../data/wisa/videos" \
    --output_dir "../../data/features/dino" \
    --ckpt_path "../../ckpt/dinov2_vitb14_reg4_pretrain.pth"
cd ..

#VGGT
cd VGGT
python fea.py \
    --video_dir "../../data/wisa/videos" \
    --output_dir "../../data/features/vggt" \
    --model_path "../../ckpt/vggt"
cd ..

#Flow
cd RAFT
python latent.py \
    --video_folder "../../data/wisa/videos" \
    --output_dir "../../data/features/flow" \
    --raft_model "../../ckpt/models/raft-things.pth" \
    --vae_checkpoint "../../ckpt/wan-t2v-1.3b-diffusers"
cd ..