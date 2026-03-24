import os
import glob
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import torch.multiprocessing as mp

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def get_args():
    parser = argparse.ArgumentParser(description="Video Feature Extraction with DINOv2 & PCA")
    parser.add_argument("--video_folder", type=str, default="/path/to/input_dir", help="Directory containing input videos")
    parser.add_argument("--output_dir", type=str, default="/path/to/output_dir", help="Directory to save extracted features")
    parser.add_argument("--repo_dir", type=str, default='./facebookresearch_dinov2_main', help="Local path to DINOv2 repository")
    parser.add_argument("--ckpt_path", type=str, default='../../ckpt/dinov2_vitb14_reg4_pretrain.pth', help="Path to DINOv2 checkpoint file")
    parser.add_argument("--pca_model_path", type=str, default="./dino_video_pca_model.pth", help="Path to the pretrained PCA model")
    parser.add_argument("--pca_rank", type=int, default=8, help="Number of PCA components to keep")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for DINOv2 inference")
    return parser.parse_args()

@torch.no_grad()
def load_dino_model(repo_dir, ckpt_path, device):
    model = torch.hub.load(repo_dir, 'dinov2_vitb14_reg', source='local', pretrained=False, verbose=False)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    model.head = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def load_pca_matrix(pca_path, rank, device):
    pca_stats = torch.load(pca_path, map_location='cpu')
    pca_model = pca_stats["pca_model"]
    
    components = torch.tensor(pca_model.components_[:rank, :], dtype=torch.float32).to(device)
    pca_mean = torch.tensor(pca_model.mean_, dtype=torch.float32).to(device)
    
    mean_dino = torch.tensor(pca_stats["mean"], dtype=torch.float32).to(device)
    std_dino = torch.tensor(pca_stats["std"], dtype=torch.float32).to(device)
    
    return components, pca_mean, mean_dino, std_dino

def preprocess_frame(frame, transform):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame)
    return transform(pil_img)

def sample_frames(video_path, num_frames=81, target_h=840, target_w=1456):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None

    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    
    frames_buffer = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            if len(frames_buffer) > 0:
                frame = frames_buffer[-1] 
            else:
                frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        frame_resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        frames_buffer.append(frame_resized)
        
    cap.release()
    return frames_buffer

@torch.no_grad()
def process_video(args, video_path, model, pca_params, transform, device):
    raw_frames = sample_frames(video_path, num_frames=81, target_h=840, target_w=1456)
    if raw_frames is None:
        raise ValueError(f"Video empty or corrupted: {video_path}")

    img_tensors = torch.stack([preprocess_frame(f, transform) for f in raw_frames])
    pca_components, pca_mean, mean_dino, std_dino = pca_params
    
    all_pca_feats = []
    num_patches_h = 840 // 14  # 60
    num_patches_w = 1456 // 14 # 104
    
    for i in range(0, len(img_tensors), args.batch_size):
        batch = img_tensors[i : i + args.batch_size].to(device, non_blocking=True)
        
        outputs = model.forward_features(batch)
        z = outputs['x_norm_patchtokens'] 
        
        # PCA projection
        z = (z - mean_dino) / std_dino
        z = z - pca_mean
        z_pca = z @ pca_components.T
        
        z_pca = z_pca.reshape(batch.shape[0], num_patches_h, num_patches_w, args.pca_rank)
        all_pca_feats.append(z_pca) 
        
    full_feats = torch.cat(all_pca_feats, dim=0) # [81, H, W, C]

    first_frame = full_feats[0:1] 
    rest_frames = full_feats[1:]  
    rest_pooled = rest_frames.view(20, 4, num_patches_h, num_patches_w, args.pca_rank).mean(dim=1) 
    final_feats = torch.cat([first_frame, rest_pooled], dim=0) # [21, H, W, C]
    
    final_feats = final_feats.permute(3, 0, 1, 2) # [C, F, H, W]
    feature_mean = final_feats.mean(dim=(1, 2, 3), keepdim=True)
    feature_std = final_feats.std(dim=(1, 2, 3), keepdim=True)
    final_feats = (final_feats - feature_mean) / (feature_std + 1e-6)
    
    return final_feats

def main_worker(rank, world_size, args, all_video_files):
    device = torch.device(f"cuda:{rank}")
    cv2.setNumThreads(0)
    
    my_files = all_video_files[rank::world_size]
    print(f"[GPU {rank}] Started. Assigned tasks: {len(my_files)}")

    try:
        model = load_dino_model(args.repo_dir, args.ckpt_path, device)
        pca_params = load_pca_matrix(args.pca_model_path, args.pca_rank, device)
    except Exception as e:
        print(f"[GPU {rank}] Model loading failed: {e}")
        return

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    iterator = tqdm(my_files, desc=f"GPU {rank}", position=rank)
    success_count = 0
    
    for video_path in iterator:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        save_path = os.path.join(args.output_dir, f"{video_name}.pth")
        
        if os.path.exists(save_path):
            continue
            
        try:
            final_tensor = process_video(args, video_path, model, pca_params, transform, device)
            save_dict = { "features": final_tensor.cpu() }
            torch.save(save_dict, save_path)
            success_count += 1
        except Exception as e:
            print(f"\n[GPU {rank}] Error processing {video_name}: {e}")

    print(f"[GPU {rank}] Finished. Successfully processed: {success_count}/{len(my_files)}")

def main():
    args = get_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv')
    all_files = []
    for ext in video_extensions:
        all_files.extend(glob.glob(os.path.join(args.video_folder, ext)))
    
    all_files = sorted(all_files)
    total_files = len(all_files)
    
    if total_files == 0:
        print("No video files found.")
        return

    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No GPU detected. Running on CPU...")
        main_worker(0, 1, args, all_files)
        return

    print(f"Detected {world_size} GPUs. Starting parallel processing for {total_files} videos...")
    print(f"Batch Size: {args.batch_size} (Increase if VRAM allows)")

    mp.spawn(
        main_worker,
        args=(world_size, args, all_files),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()