import os
import glob
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from einops import rearrange
import sys
import argparse 
from decord import VideoReader, cpu

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from models.vggt import VGGT
except ImportError:
    pass

cv2.setNumThreads(0) 

class VGGTFolderProcessor:
    def __init__(self, model_path, device):
        self.device = device
        self.patch_size = 14 
        
        print(f"[Rank {device}] Loading VGGT model from local: {model_path} ...")
        
        try:
            self.model = VGGT.from_pretrained(model_path, use_safetensors=False)
        except TypeError:
            self.model = VGGT.from_pretrained(model_path)
            
        self.model.eval()
        self.model.to(self.device)
        print(f"[Rank {device}] Model loaded successfully.")

    def preprocess_frames(self, frames_list, target_h=420, target_w=728):
        frames = np.stack(frames_list)
        frames = torch.from_numpy(frames).float() / 255.0
        frames = rearrange(frames, 't h w c -> t c h w')
        frames = F.interpolate(frames, size=(target_h, target_w), mode='bilinear', align_corners=False)

        frames = frames.unsqueeze(0) 
        return frames.to(self.device)

    def temporal_compression(self, feats):
        """
        Input: [C, 81, H, W] 
        Output: [C, 21, H, W]
        """
        first_frame = feats[:, 0:1, :, :]  
        rest_frames = feats[:, 1:, :, :]   
        rest_frames = rest_frames.unsqueeze(0) 
        
        target_t = 20
        target_h, target_w = feats.shape[2], feats.shape[3]
        
        pooled_rest = F.adaptive_avg_pool3d(
            rest_frames, 
            output_size=(target_t, target_h, target_w)
        ) 
        
        pooled_rest = pooled_rest.squeeze(0) 
        final_output = torch.cat([first_frame, pooled_rest], dim=1) 
        
        return final_output

    def normalize_features(self, feat):
        """
        Feature Normalization: (feat - mean) / std
        Input: [C, T, H, W]
        """
        reduce_dims = (1, 2, 3) # T, H, W
        
        feature_mean = feat.mean(dim=reduce_dims, keepdim=True)
        feature_std = feat.std(dim=reduce_dims, keepdim=True)
        
        normalized_feat = (feat - feature_mean) / (feature_std + 1e-6)
        return normalized_feat

    def extract_and_save(self, video_path, output_folder, num_frames=81, input_h=420, input_w=728, target_h=60, target_w=104):
        video_path = Path(video_path)
        output_path = Path(output_folder) / f"{video_path.stem}.pth"
        
        if output_path.exists():
            return

        frames = self._sample_frames(video_path, num_frames)
        if len(frames) != num_frames:
            print(f"Error: Video {video_path} frames mismatch. Got {len(frames)}, expected {num_frames}.")
            return
        input_tensor = self.preprocess_frames(frames, target_h=input_h, target_w=input_w)
        B, T, C, H, W = input_tensor.shape # [1, 81, 3, 420, 728]

        with torch.no_grad():
            aggregated_tokens_list, _ = self.model.shortcut_forward(input_tensor)
            raw_feats = aggregated_tokens_list[-1] 
            
            if raw_feats.dim() == 4:
                raw_feats = rearrange(raw_feats, 'b t n c -> (b t) n c')
            spatial_feats = raw_feats[:, 5:, :] 

            feat_h = H // self.patch_size # 420 / 14 = 30
            feat_w = W // self.patch_size # 728 / 14 = 52

            spatial_feats = rearrange(spatial_feats, 'bt (h w) c -> bt c h w', h=feat_h, w=feat_w)

            resized_feats = F.interpolate(
                spatial_feats, 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            )

            final_feats = rearrange(resized_feats, '(b t) c h w -> b t c h w', b=B, t=T)
            final_feats = final_feats.squeeze(0) # [T, C, 60, 104]

            final_feats = rearrange(final_feats, 't c h w -> c t h w')

            compressed_feats = self.temporal_compression(final_feats)

            normalized_feats = self.normalize_features(compressed_feats)

        save_dict = {"features": normalized_feats.half().cpu()}
        torch.save(save_dict, output_path)

    def _sample_frames(self, video_path, num_frames):
        try:
            vr = VideoReader(str(video_path), ctx=cpu(0))
            total_frames = len(vr)
            
            if total_frames == 0:
                print(f"Warning: Empty video {video_path}")
                return []

            indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
            frames = vr.get_batch(indices).asnumpy()
            return list(frames)
            
        except Exception as e:
            print(f"Error reading video {video_path}: {e}")
            return []

def process_all_videos(input_folder, output_folder, model_path):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    if rank == 0:
        output_path.mkdir(parents=True, exist_ok=True)
    
    video_files = sorted(list(input_path.glob("*.mp4")))
    
    if not video_files:
        print(f"No .mp4 files found in {input_folder}")
        return

    my_files = video_files[rank::world_size]
    print(f"[Rank {rank}/{world_size}] Processing {len(my_files)} videos on {device}")

    processor = VGGTFolderProcessor(model_path=model_path, device=device)
    
    for video_file in tqdm(my_files, desc=f"Rank {rank}", position=rank):
        try:
            processor.extract_and_save(
                video_file, 
                output_folder, 
                num_frames=81,     
                input_h=420,       
                input_w=728,       
                target_h=60,        
                target_w=104       
            )
        except Exception as e:
            print(f"[Rank {rank}] Failed {video_file.name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract VGGT features from a folder of videos.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing input videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted features")
    parser.add_argument("--model_path", type=str, required=True, help="Local path to the VGGT model")
    
    args = parser.parse_args()
    
    process_all_videos(args.video_dir, args.output_dir, args.model_path)