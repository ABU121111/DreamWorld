import sys
import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from typing import List, Optional, Dict, Any
import inspect
from tqdm import tqdm
from raft import RAFT
from utils.utils import InputPadder
from diffusers import AutoencoderKLWan

sys.path.append('core')

class ProcessorMixin:
    def __init__(self) -> None:
        self._forward_parameter_names = inspect.signature(self.forward).parameters.keys()
        self.output_names: List[str] = None
        self.input_names: Dict[str, Any] = None

    def __call__(self, *args, **kwargs) -> Any:
        shallow_copy_kwargs = dict(kwargs.items())
        if self.input_names is not None:
            for k, v in self.input_names.items():
                if k in shallow_copy_kwargs:
                    shallow_copy_kwargs[v] = shallow_copy_kwargs.pop(k)
        acceptable_kwargs = {k: v for k, v in shallow_copy_kwargs.items() if k in self._forward_parameter_names}
        output = self.forward(*args, **acceptable_kwargs)
        if "__drop__" in output:
            output.pop("__drop__")
        return output

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("ProcessorMixin::forward method should be implemented by the subclass.")

class WanLatentEncodeProcessor(ProcessorMixin):
    def __init__(self, output_names: List[str]):
        super().__init__()
        self.output_names = output_names
        assert len(self.output_names) == 3

    def forward(
        self,
        vae: AutoencoderKLWan,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        device = vae.device
        dtype = vae.dtype
        if image is not None:
            video = image.unsqueeze(1)
        assert video.ndim == 5, f"Expected 5D tensor, got {video.ndim}D tensor"

        video = video.to(device=device, dtype=dtype)
        video = video.permute(0, 2, 1, 3, 4).contiguous()

        if compute_posterior:
            latents = vae.encode(video).latent_dist.sample(generator=generator)
            latents = latents.to(dtype=dtype)
        else:
            moments = vae._encode(video)
            latents = moments.to(dtype=dtype)

        latents_mean = torch.tensor(vae.config.latents_mean).to(device, dtype=dtype)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).to(device, dtype=dtype)

        return {self.output_names[0]: latents, self.output_names[1]: latents_mean, self.output_names[2]: latents_std}

def flow_to_videojam_rgb(flow):
    h, w, _ = flow.shape
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    diag = np.sqrt(h**2 + w**2)
    sigma = 0.15
    rho = np.sqrt(u**2 + v**2)
    norm_factor = sigma * diag + 1e-6
    m = rho / norm_factor
    m = np.minimum(1.0, m)
    angle = np.arctan2(v, u)
    hsv = np.zeros((h, w, 3), dtype=np.float32)
    angle_deg = (angle * 180 / np.pi) + 180
    hsv[:, :, 0] = angle_deg / 2.0
    hsv[:, :, 1] = m * 255.0
    hsv[:, :, 2] = 255.0
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return bgr

def load_video_frames(video_path, device, target_frames=81, width=832, height=480):
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    num_frames = len(all_frames)
    if num_frames < 2:
        return None

    indices = torch.linspace(0, num_frames - 1, target_frames).long()

    frames_buffer = []
    for idx in indices:
        frame = all_frames[idx.item()]
        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().to(device)
        frames_buffer.append(frame_tensor)
    return frames_buffer

def compute_flow_batch(raft_model, frames_buffer, batch_size=16, device='cuda'):
    flows = []
    num_pairs = len(frames_buffer) - 1

    for start_idx in range(0, num_pairs, batch_size):
        end_idx = min(start_idx + batch_size, num_pairs)
        actual_batch_size = end_idx - start_idx

        image1_batch = torch.stack(frames_buffer[start_idx:end_idx], dim=0).to(device)
        image2_batch = torch.stack(frames_buffer[start_idx + 1:end_idx + 1], dim=0).to(device)

        padder = InputPadder(image1_batch.shape)
        image1_batch, image2_batch = padder.pad(image1_batch, image2_batch)

        with torch.no_grad():
            _, flow_up_batch = raft_model(image1_batch, image2_batch, iters=20, test_mode=True)

        flow_up_batch = flow_up_batch.cpu()
        for i in range(actual_batch_size):
            flow = flow_up_batch[i].permute(1, 2, 0).numpy()
            flows.append(flow)
    return flows

def process_single_video(video_path, output_dir, raft_model, vae_model, vae_processor, device):
    video_name = os.path.basename(video_path)
    file_stem = os.path.splitext(video_name)[0]
    save_path = os.path.join(output_dir, f"{file_stem}.pt")

    if os.path.exists(save_path):
        print(f"Skipping {video_name}, already exists.")
        return

    frames_buffer = load_video_frames(video_path, device=device, target_frames=81, width=832, height=480)
    if frames_buffer is None:
        print(f"Error: Could not load frames for {video_name}")
        return

    flow_rgb_list = []

    with torch.no_grad():
        for i in range(len(frames_buffer) - 1):
            image1 = frames_buffer[i][None]
            image2 = frames_buffer[i + 1][None]

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_up = raft_model(image1, image2, iters=20, test_mode=True)
            flo = flow_up[0].permute(1, 2, 0).cpu().numpy()

            bgr_flow_img = flow_to_videojam_rgb(flo)
            rgb_flow_img = cv2.cvtColor(bgr_flow_img, cv2.COLOR_BGR2RGB)

            flow_rgb_list.append(rgb_flow_img)

    if len(flow_rgb_list) > 0:
        first_frame = flow_rgb_list[0].copy()
        flow_rgb_list.insert(0, first_frame)
        assert len(flow_rgb_list) == 81, f"Expected 81 flow frames, got {len(flow_rgb_list)}"

    flow_tensor = torch.tensor(np.array(flow_rgb_list), dtype=vae_model.dtype, device=device)
    flow_tensor = flow_tensor.permute(0, 3, 1, 2)
    flow_tensor = (flow_tensor / 255.0 - 0.5) * 2.0
    vae_input = flow_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = vae_processor.forward(
            vae=vae_model,
            video=vae_input,
            compute_posterior=True
        )

    raw_latents = outputs["latents"]
    mean = outputs["latents_mean"]
    std = outputs["latents_std"]

    if mean.ndim == 1:
        mean = mean.view(1, -1, 1, 1, 1)
        std = std.view(1, -1, 1, 1, 1)

    final_latents = (raw_latents.float() - mean.float()) * std.float()
    final_latents = final_latents.to(raw_latents.dtype)

    torch.save(final_latents.cpu(), save_path)

def worker_main(rank, world_size, args, video_files):
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    print(f"[Rank {rank}] Using device {device}")

    raft = RAFT(args)
    state_dict = torch.load(args.raft_model, map_location='cpu', weights_only=True)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    raft.load_state_dict(new_state_dict)
    raft.to(device)
    raft.eval()

    vae = AutoencoderKLWan.from_pretrained(args.vae_checkpoint, subfolder="vae")
    vae = vae.to(device)
    if torch.cuda.is_bf16_supported():
        vae = vae.to(torch.bfloat16)
    else:
        vae = vae.to(torch.float16)
    vae.eval()

    processor = WanLatentEncodeProcessor(output_names=["latents", "latents_mean", "latents_std"])

    my_files = video_files[rank::world_size]
    print(f"[Rank {rank}] Assigned {len(my_files)} videos")

    for video_path in tqdm(my_files, desc=f"[Rank {rank}] Processing videos"):
        try:
            process_single_video(video_path, args.output_dir, raft, vae, processor, device)
        except Exception as e:
            print(f"[Rank {rank}] Error processing {video_path}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raft_model', default="../../ckpt/models/raft-things.pth", help="Restore RAFT checkpoint")
    parser.add_argument('--small', action='store_true', help='Use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='Use efficient correlation implementation')

    parser.add_argument('--video_folder', default="/path/to/input_dir", help="Input video folder")
    parser.add_argument('--output_dir', default="/path/to/output_dir", help="Output folder for latents")

    parser.add_argument('--vae_checkpoint', default="/path/to/Wan2.1-T2V-1.3B-Diffusers", help="VAE checkpoint or repo ID")

    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(), help="Number of GPUs to use")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [
        os.path.join(args.video_folder, f)
        for f in os.listdir(args.video_folder)
        if f.lower().endswith(video_extensions)
    ]
    video_files.sort()

    print(f"Total videos found: {len(video_files)}")
    print(f"Spawning {args.num_gpus} processes...")

    mp.spawn(
        worker_main,
        args=(args.num_gpus, args, video_files),
        nprocs=args.num_gpus,
        join=True
    )

    print("All processing done.")

if __name__ == '__main__':
    main()