# dataset
huggingface-cli download --repo-type dataset qihoo360/WISA-80K \
--local-dir ./data/wisa \
--include "data/wisa-80k.json"

huggingface-cli download --repo-type dataset qihoo360/WISA-80K \
--local-dir ./data/wisa/videos \
--include "data/videos/2[0-9].zip"

huggingface-cli download --repo-type dataset qihoo360/WISA-80K \
--local-dir ./data/wisa/videos \
--include "data/videos/3[0-9].zip"

huggingface-cli download --repo-type dataset qihoo360/WISA-80K \
--local-dir ./data/wisa/videos \
--include "data/videos/4[0-9].zip"

# model
# Wan-T2V-1.3B-Diffusers
huggingface-cli download --repo-type model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
--local-dir ./ckpt/wan2.1-t2v-1.3b-diffusers

# VGGT
huggingface-cli download --repo-type model facebook/VGGT-1B \
--local-dir ./ckpt/vggt

# DINOv2
wget -P ./ckpt/ https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth

#RAFT
wget -P ./ckpt/ https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
cd ckpt
unzip models.zip
rm -rf models.zip
cd ..