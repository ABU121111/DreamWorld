import os
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Sequence, Union, List, Optional, Tuple, Dict

import torch
import torch.nn as nn
from safetensors.torch import load_file
from diffusers import WanPipeline
from diffusers.pipelines.wan.pipeline_wan import WanPipelineOutput


class CustomWanPipeline:
    def __init__(self, base_pipeline: WanPipeline):
        self._base_pipeline = base_pipeline
        
    def __getattr__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            if hasattr(self._base_pipeline, name):
                return getattr(self._base_pipeline, name)
            else:
                raise AttributeError(f"'{type(self).__name__}' and '{type(self._base_pipeline).__name__}' object has no attribute '{name}'")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        base_pipeline = WanPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(base_pipeline)

    def to(self, device):
        self._base_pipeline.to(device)
        return self

    def load_lora_weights(self, pretrained_model_name_or_path_or_dict, adapter_name="default", **kwargs):
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            state_dict = pretrained_model_name_or_path_or_dict
        else:
            weights_path = pretrained_model_name_or_path_or_dict
            if os.path.isdir(weights_path):
                weights_path = os.path.join(weights_path, "pytorch_lora_weights.safetensors")
            
            if not os.path.exists(weights_path):
                 print(f"Warning: Could not find local file at {weights_path}, delegating to standard loader.")
                 return self._base_pipeline.load_lora_weights(pretrained_model_name_or_path_or_dict, **kwargs)
            
            state_dict = load_file(weights_path)

        custom_layer_keys = {}
        lora_keys = {}
        
        target_modules = ["patch_embedding", "proj_out"]
        
        for k, v in state_dict.items():
            is_custom = any(m in k for m in target_modules) and "lora" not in k
            if is_custom:
                clean_key = k.replace("transformer.", "")
                custom_layer_keys[clean_key] = v
            else:
                lora_keys[k] = v

        if "patch_embedding.weight" in custom_layer_keys:
            new_weight = custom_layer_keys["patch_embedding.weight"]
            new_in_channels = new_weight.shape[1] 
            
            current_layer = self._base_pipeline.transformer.patch_embedding
            
            if new_in_channels != current_layer.in_channels:
                print(f"[CustomWanPipeline] Resizing patch_embedding: {current_layer.in_channels} -> {new_in_channels} channels")
                
                new_layer = nn.Conv3d(
                    in_channels=new_in_channels,
                    out_channels=current_layer.out_channels,
                    kernel_size=current_layer.kernel_size,
                    stride=current_layer.stride,
                    padding=current_layer.padding,
                ).to(dtype=self._base_pipeline.transformer.dtype, device=self._base_pipeline.device)
                
                self._base_pipeline.transformer.patch_embedding = new_layer

        if "proj_out.weight" in custom_layer_keys:
            new_weight = custom_layer_keys["proj_out.weight"]
            new_out_features = new_weight.shape[0]
            
            current_layer = self._base_pipeline.transformer.proj_out
            
            if isinstance(current_layer, nn.Linear) and new_out_features != current_layer.out_features:
                print(f"[CustomWanPipeline] Resizing proj_out: {current_layer.out_features} -> {new_out_features} features")
                
                new_layer = nn.Linear(
                    in_features=current_layer.in_features,
                    out_features=new_out_features,
                    bias=True if custom_layer_keys.get("proj_out.bias") is not None else False
                ).to(dtype=self._base_pipeline.transformer.dtype, device=self._base_pipeline.device)
                
                self._base_pipeline.transformer.proj_out = new_layer

        if custom_layer_keys:
            print(f"[CustomWanPipeline] Loading full weights for: {list(custom_layer_keys.keys())}")
            self._base_pipeline.transformer.load_state_dict(custom_layer_keys, strict=False)

        if lora_keys:
            print(f"[CustomWanPipeline] Loading {len(lora_keys)} LoRA adapters...")
            self._base_pipeline.load_lora_weights(lora_keys, adapter_name=adapter_name, **kwargs)
        else:
            print("[CustomWanPipeline] No LoRA keys found in checkpoint.")
        
        return self


    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            negative_prompt: Union[str, List[str]] = None,
            height: int = 480,
            width: int = 832,
            num_frames: int = 81,
            num_inference_steps: int = 50,
            guidance_scale: float = 5,
            dino_guidance_scale: float = 1,
            vggt_guidance_scale: float = 1,
            flow_guidance_scale: float = 1,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            output_type: Optional[str] = "np",
            return_dict: bool = True,
            attention_kwargs: Optional[Dict[str, Any]] = None,

            dino_generator: Optional[torch.Generator] = None,
            vggt_generator: Optional[torch.Generator] = None,
            flow_generator: Optional[torch.Generator] = None,
            enable_inner_guidance: bool = True,
            **kwargs,
    ):
        self._base_pipeline._guidance_scale = guidance_scale
        self._base_pipeline._attention_kwargs = attention_kwargs

        self.check_inputs(prompt, negative_prompt, height, width, prompt_embeds, negative_prompt_embeds)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        transformer_dtype = self.transformer.dtype

        # 1. Config Handling for Quad-Stream
        try:
            vae_out_ch = self.transformer.config.original_out_channels
            vae_in_ch = self.transformer.config.original_in_channels

            dino_in_ch = self.transformer.config.dino_in_channels
            dino_out_ch = self.transformer.config.dino_out_channels

            vggt_in_ch = self.transformer.config.vggt_in_channels
            vggt_out_ch = self.transformer.config.vggt_out_channels

            flow_in_ch = self.transformer.config.flow_in_channels
            flow_out_ch = self.transformer.config.flow_out_channels
        except AttributeError as e:
            raise AttributeError(
                "Transformer config is missing quad-stream attributes. "
                f"Error: {e}"
            )

        # 2. Encode Prompts
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=device,
        )
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 3. Construct Input Prompts for Guidance
        # Order: [Joint, Uncond, No_DINO, No_VGGT, No_FLOW]
        # Text : [Pos,   Neg,    Pos,     Pos,     Pos   ]
        if enable_inner_guidance:
            prompt_embeds_input = torch.cat([
                prompt_embeds,  # 1. Joint
                negative_prompt_embeds,  # 2. Uncond
                prompt_embeds,  # 3. No_DINO
                prompt_embeds,  # 4. No_VGGT
                prompt_embeds  # 5. No_FLOW
            ])
        else:
            # Standard CFG: [Pos, Neg]
            prompt_embeds_input = torch.cat([prompt_embeds, negative_prompt_embeds])

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        # 4. Prepare Latents for all streams
        # 4.1 VAE Noise
        latents_vae = self.prepare_latents(
            batch_size * num_videos_per_prompt, vae_out_ch, height, width,
            num_frames, transformer_dtype, device, generator, None,
        )

        # 4.2 DINO Noise
        if dino_generator is None:
            dino_generator = torch.Generator(device=device)
            dino_generator.manual_seed(generator.initial_seed() + 1 if generator else 42)
        latents_dino = self.prepare_latents(
            batch_size * num_videos_per_prompt, dino_in_ch, height, width,
            num_frames, transformer_dtype, device, dino_generator, None,
        )

        # 4.3 VGGT Noise
        if vggt_generator is None:
            vggt_generator = torch.Generator(device=device)
            vggt_generator.manual_seed(generator.initial_seed() + 2 if generator else 43)
        latents_vggt = self.prepare_latents(
            batch_size * num_videos_per_prompt, vggt_in_ch, height, width,
            num_frames, transformer_dtype, device, vggt_generator, None,
        )

        # 4.4 FLOW Noise
        if flow_generator is None:
            flow_generator = torch.Generator(device=device)
            flow_generator.manual_seed(generator.initial_seed() + 3 if generator else 44)
        latents_flow = self.prepare_latents(
            batch_size * num_videos_per_prompt, flow_in_ch, height, width,
            num_frames, transformer_dtype, device, flow_generator, None,
        )

        # 4.5 Concatenate All Streams
        # Order MUST match Model Input: [VAE, DINO, VGGT, FLOW]
        latents = torch.cat([latents_vae, latents_dino, latents_vggt, latents_flow], dim=1)

        if isinstance(num_inference_steps, torch.Tensor):
            num_inference_steps = int(num_inference_steps.item())

        # Define indices for masking logic
        idx_vae_end = vae_out_ch
        idx_dino_end = idx_vae_end + dino_in_ch
        idx_vggt_end = idx_dino_end + vggt_in_ch
        idx_flow_end = idx_vggt_end + flow_in_ch

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                current_latents = latents.to(device=device, dtype=transformer_dtype)

                if enable_inner_guidance:
                    # 5. Inner Guidance Input Construction
                    # 1. Joint: All Active
                    # 2. Uncond: All Active (but negative prompt)
                    batch_joint_uncond = torch.cat([current_latents, current_latents])

                    # 3. No-DINO: Mask DINO part
                    latents_no_dino = current_latents.clone()
                    latents_no_dino[:, idx_vae_end:idx_dino_end, ...] = 0.0

                    # 4. No-VGGT: Mask VGGT part
                    latents_no_vggt = current_latents.clone()
                    latents_no_vggt[:, idx_dino_end:idx_vggt_end, ...] = 0.0

                    # 5. No-FLOW: Mask FLOW part
                    latents_no_flow = current_latents.clone()
                    latents_no_flow[:, idx_vggt_end:idx_flow_end, ...] = 0.0

                    # Concatenate batch [5*B]
                    latent_model_input = torch.cat([
                        batch_joint_uncond,
                        latents_no_dino,
                        latents_no_vggt,
                        latents_no_flow
                    ])
                else:
                    # Standard CFG [2*B]
                    latent_model_input = torch.cat([current_latents] * 2)

                timestep = t.expand(latent_model_input.shape[0])

                # 6. Forward Pass
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds_input,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                # 7. Guidance Calculation
                if enable_inner_guidance:
                    # Chunk into 5 parts
                    pred_joint, pred_uncond, pred_no_dino, pred_no_vggt, pred_no_flow = noise_pred.chunk(5)

                    # Calculate Guidance Vectors
                    text_guidance = pred_joint - pred_uncond
                    dino_guidance = pred_joint - pred_no_dino
                    vggt_guidance = pred_joint - pred_no_vggt
                    flow_guidance = pred_joint - pred_no_flow

                    # Apply Scales
                    final_pred = pred_joint + \
                                 (guidance_scale * text_guidance) + \
                                 (dino_guidance_scale * dino_guidance) + \
                                 (vggt_guidance_scale * vggt_guidance) + \
                                 (flow_guidance_scale * flow_guidance)
                else:
                    pred_cond, pred_uncond = noise_pred.chunk(2)
                    final_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

                # 8. Output Processing & Scheduler Step
                # Output Split: [VAE, DINO, VGGT, FLOW]
                pred_vae_final, pred_dino_final, pred_vggt_final, pred_flow_final = torch.split(
                    final_pred,
                    [vae_out_ch, dino_out_ch, vggt_out_ch, flow_out_ch],
                    dim=1
                )

                # Reconstruct full input for scheduler step (Must match input structure)
                full_pred_for_step = torch.cat([
                    pred_vae_final,
                    pred_dino_final,
                    pred_vggt_final,
                    pred_flow_final
                ], dim=1)
                
                # Step
                latents = self.scheduler.step(full_pred_for_step, t, current_latents, return_dict=False)[0]

                progress_bar.update()

        # 9. Decode Video (VAE only)
        if not output_type == "latent":
            # Extract VAE part only
            video_latents = latents[:, :vae_out_ch, :, :, :].to(self.vae.dtype)
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                video_latents.device, video_latents.dtype)
            latents_std = torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                video_latents.device, video_latents.dtype)
            video_latents = video_latents * latents_std + latents_mean
            video = self.vae.decode(video_latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)