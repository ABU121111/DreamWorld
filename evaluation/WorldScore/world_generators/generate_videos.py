import os
import sys
import math
import json
import glob
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Sequence, Union, List, Optional, Dict

import fire
import hydra
import omegaconf
import structlog
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from PIL import Image

from safetensors.torch import load_file
from diffusers import WanPipeline, WanImageToVideoPipeline
from diffusers import WanTransformer3DModel as OriginalWanTransformer3DModel 
from diffusers.pipelines.wan.pipeline_wan import WanPipelineOutput

from worldscore.benchmark.helpers import GetHelpers
from worldscore.benchmark.utils.utils import check_model, type2model, get_model2type

logger = structlog.getLogger()

class QuadStreamTransformer3DModel(OriginalWanTransformer3DModel):
    def __init__(
        self,
        dino_in_channels: int = 0,
        dino_out_channels: int = 0,
        vggt_in_channels: int = 0,
        vggt_out_channels: int = 0,
        flow_in_channels: int = 0,
        flow_out_channels: int = 0,
        **kwargs,
    ):
        if "original_in_channels" in kwargs:
            original_in_channels = kwargs.pop("original_in_channels")
            original_out_channels = kwargs.pop("original_out_channels")
            new_in_channels = kwargs.pop("in_channels")
            new_out_channels = kwargs.pop("out_channels")
            
            super().__init__(
                in_channels=new_in_channels, 
                out_channels=new_out_channels, 
                **kwargs
            )
        else:
            original_in_channels = kwargs.pop("in_channels")
            original_out_channels = kwargs.pop("out_channels")

            new_in_channels = original_in_channels + dino_in_channels + vggt_in_channels + flow_in_channels
            new_out_channels = original_out_channels + dino_out_channels + vggt_out_channels + flow_out_channels

            super().__init__(
                in_channels=new_in_channels, 
                out_channels=new_out_channels, 
                **kwargs
            )

        config_dict = dict(self.config)
        config_dict['dino_in_channels'] = dino_in_channels
        config_dict['dino_out_channels'] = dino_out_channels
        config_dict['vggt_in_channels'] = vggt_in_channels
        config_dict['vggt_out_channels'] = vggt_out_channels
        config_dict['flow_in_channels'] = flow_in_channels
        config_dict['flow_out_channels'] = flow_out_channels
        config_dict['original_in_channels'] = original_in_channels
        config_dict['original_out_channels'] = original_out_channels
        self.register_to_config(**config_dict)


class CustomWanPipeline:
    def __init__(self, base_pipeline: Union[WanPipeline, WanImageToVideoPipeline]):
        self._base_pipeline = base_pipeline

    def __getattr__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            if hasattr(self._base_pipeline, name):
                return getattr(self._base_pipeline, name)
            else:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'"
                )

    def to(self, device):
        self._base_pipeline.to(device)
        return self

    def load_lora_weights(self, pretrained_model_name_or_path_or_dict, adapter_name="default", **kwargs):
        logger.info(f"Loading weights from {pretrained_model_name_or_path_or_dict}")
        
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            state_dict = pretrained_model_name_or_path_or_dict
        else:
            weights_path = pretrained_model_name_or_path_or_dict
            if os.path.isdir(weights_path):
                weights_path = os.path.join(weights_path, "pytorch_lora_weights.safetensors")
            
            if not os.path.exists(weights_path):
                logger.warning(f"File {weights_path} not found. Delegating to standard loader.")
                return self._base_pipeline.load_lora_weights(pretrained_model_name_or_path_or_dict, **kwargs)

            state_dict = load_file(weights_path)

        custom_layer_keys = {}
        lora_keys = {}
        full_weight_keywords = ["patch_embedding", "proj_out"]

        for k, v in state_dict.items():
            is_full_weight = any(m in k for m in full_weight_keywords) and "lora" not in k
            if is_full_weight:
                clean_key = k.replace("transformer.", "") if k.startswith("transformer.") else k
                custom_layer_keys[clean_key] = v
            else:
                lora_keys[k] = v

        if custom_layer_keys:
            self._base_pipeline.transformer.load_state_dict(custom_layer_keys, strict=False)

        if lora_keys:
            try:
                self._base_pipeline.load_lora_weights(lora_keys, adapter_name=adapter_name, **kwargs)
            except Exception as e:
                logger.error(f"Failed to load LoRA adapters: {e}")
        
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
        guidance_scale: float = 5.0,
        dino_guidance_scale: float = 1.0, 
        vggt_guidance_scale: float = 1.0, 
        flow_guidance_scale: float = 1.0, 
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None, 
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        image: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
        dino_generator: Optional[torch.Generator] = None,
        vggt_generator: Optional[torch.Generator] = None,
        flow_generator: Optional[torch.Generator] = None,
        enable_inner_guidance: bool = True, 
        **kwargs,
    ):
        pipeline = self._base_pipeline
        pipeline._guidance_scale = guidance_scale
        pipeline._attention_kwargs = attention_kwargs

        pipeline.check_inputs(prompt, negative_prompt, height, width, prompt_embeds, negative_prompt_embeds)

        if prompt is not None and isinstance(prompt, str): 
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list): 
            batch_size = len(prompt)
        else: 
            batch_size = prompt_embeds.shape[0]
        
        device = pipeline._execution_device
        transformer_dtype = pipeline.transformer.dtype

        try:
            vae_out_ch = pipeline.transformer.config.original_out_channels
            vae_in_ch = pipeline.transformer.config.original_in_channels
            dino_in_ch = pipeline.transformer.config.dino_in_channels
            dino_out_ch = pipeline.transformer.config.dino_out_channels
            vggt_in_ch = pipeline.transformer.config.vggt_in_channels
            vggt_out_ch = pipeline.transformer.config.vggt_out_channels
            flow_in_ch = pipeline.transformer.config.flow_in_channels
            flow_out_ch = pipeline.transformer.config.flow_out_channels
        except AttributeError as e:
            raise AttributeError(f"Missing quad-stream attributes in config: {e}")
            
        is_i2v = "image_dim" in pipeline.transformer.config and pipeline.transformer.config.image_dim is not None
        i2v_ch = vae_in_ch - vae_out_ch if is_i2v else 0

        prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
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

        if enable_inner_guidance:
            prompt_embeds_input = torch.cat([
                prompt_embeds,           
                negative_prompt_embeds,  
                prompt_embeds,           
                prompt_embeds,           
                prompt_embeds            
            ])
        else:
            prompt_embeds_input = torch.cat([prompt_embeds, negative_prompt_embeds])

        pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipeline.scheduler.timesteps

        i2v_conditioning = None
        if is_i2v and image is not None:
            i2v_latents, i2v_mask = pipeline.prepare_image_latents(
                 image=image,
                 last_image=last_image,
                 batch_size=batch_size * num_videos_per_prompt,
                 num_frames=num_frames,
                 height=height,
                 width=width,
                 generator=generator,
                 dtype=transformer_dtype,
                 device=device
            )
            i2v_conditioning = torch.cat([i2v_latents, i2v_mask], dim=1)
        
        latents_vae = pipeline.prepare_latents(
            batch_size * num_videos_per_prompt, vae_out_ch, height, width,
            num_frames, transformer_dtype, device, generator, None,
        )
        
        if dino_generator is None:
            dino_generator = torch.Generator(device=device)
            dino_generator.manual_seed(generator.initial_seed() + 1 if generator else 42)
        latents_dino = pipeline.prepare_latents(
            batch_size * num_videos_per_prompt, dino_in_ch, height, width,
            num_frames, transformer_dtype, device, dino_generator, None,
        )

        if vggt_generator is None:
            vggt_generator = torch.Generator(device=device)
            vggt_generator.manual_seed(generator.initial_seed() + 2 if generator else 43)
        latents_vggt = pipeline.prepare_latents(
            batch_size * num_videos_per_prompt, vggt_in_ch, height, width,
            num_frames, transformer_dtype, device, vggt_generator, None,
        )

        if flow_generator is None:
            flow_generator = torch.Generator(device=device)
            flow_generator.manual_seed(generator.initial_seed() + 3 if generator else 44)
        latents_flow = pipeline.prepare_latents(
            batch_size * num_videos_per_prompt, flow_in_ch, height, width,
            num_frames, transformer_dtype, device, flow_generator, None,
        )

        if is_i2v and i2v_conditioning is not None:
            latents = torch.cat([latents_vae, i2v_conditioning, latents_dino, latents_vggt, latents_flow], dim=1)
        else:
            latents = torch.cat([latents_vae, latents_dino, latents_vggt, latents_flow], dim=1)

        if isinstance(num_inference_steps, torch.Tensor):
            num_inference_steps = int(num_inference_steps.item())
        
        idx_vae_end = vae_out_ch
        idx_i2v_end = idx_vae_end + (i2v_ch if is_i2v and i2v_conditioning is not None else 0)
        idx_dino_end = idx_i2v_end + dino_in_ch
        idx_vggt_end = idx_dino_end + vggt_in_ch
        idx_flow_end = idx_vggt_end + flow_in_ch

        with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                current_latents = latents.to(device=device, dtype=transformer_dtype)
                
                if enable_inner_guidance:
                    batch_joint_uncond = torch.cat([current_latents, current_latents])
                    
                    latents_no_dino = current_latents.clone()
                    latents_no_dino[:, idx_i2v_end : idx_dino_end, ...] = 0.0
                    
                    latents_no_vggt = current_latents.clone()
                    latents_no_vggt[:, idx_dino_end : idx_vggt_end, ...] = 0.0

                    latents_no_flow = current_latents.clone()
                    latents_no_flow[:, idx_vggt_end : idx_flow_end, ...] = 0.0
                    
                    latent_model_input = torch.cat([
                        batch_joint_uncond, 
                        latents_no_dino, 
                        latents_no_vggt, 
                        latents_no_flow
                    ])
                else:
                    latent_model_input = torch.cat([current_latents] * 2)

                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = pipeline.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds_input,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                if enable_inner_guidance:
                    pred_joint, pred_uncond, pred_no_dino, pred_no_vggt, pred_no_flow = noise_pred.chunk(5)
                    
                    text_guidance = pred_joint - pred_uncond
                    dino_guidance = pred_joint - pred_no_dino
                    vggt_guidance = pred_joint - pred_no_vggt
                    flow_guidance  = pred_joint - pred_no_flow
                    
                    final_pred = pred_joint + \
                                 (guidance_scale * text_guidance) + \
                                 (dino_guidance_scale * dino_guidance) + \
                                 (vggt_guidance_scale * vggt_guidance) + \
                                 (flow_guidance_scale * flow_guidance)
                else:
                    pred_cond, pred_uncond = noise_pred.chunk(2)
                    final_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

                pred_vae_final, pred_dino_final, pred_vggt_final, pred_flow_final = torch.split(
                    final_pred, 
                    [vae_out_ch, dino_out_ch, vggt_out_ch, flow_out_ch], 
                    dim=1
                )
                
                if is_i2v and i2v_conditioning is not None:
                    i2v_current = current_latents[:, idx_vae_end : idx_i2v_end, ...]
                    full_pred_for_step = torch.cat([
                        pred_vae_final, i2v_current, pred_dino_final, pred_vggt_final, pred_flow_final
                    ], dim=1)
                else:
                    full_pred_for_step = torch.cat([
                        pred_vae_final, pred_dino_final, pred_vggt_final, pred_flow_final
                    ], dim=1)
                
                latents = pipeline.scheduler.step(full_pred_for_step, t, current_latents, return_dict=False)[0]
                progress_bar.update()

        if not output_type == "latent":
            video_latents = latents[:, :vae_out_ch, :, :, :].to(pipeline.vae.dtype) 
            latents_mean = torch.tensor(pipeline.vae.config.latents_mean).view(1, pipeline.vae.config.z_dim, 1, 1, 1).to(video_latents.device, video_latents.dtype)
            latents_std = torch.tensor(pipeline.vae.config.latents_std).view(1, pipeline.vae.config.z_dim, 1, 1, 1).to(video_latents.device, video_latents.dtype)
            video_latents = video_latents * latents_std + latents_mean
            video = pipeline.vae.decode(video_latents, return_dict=False)[0]
            video = pipeline.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        if not return_dict:
            return (video,)
        return WanPipelineOutput(frames=video)

def build_custom_transformer(pretrained_model_name_or_path, dtype, dino_channels=16):
    transformer_path = os.path.join(pretrained_model_name_or_path, "transformer")
    original_config = OriginalWanTransformer3DModel.load_config(transformer_path)
    new_config = original_config.copy()
    new_config['dino_in_channels'] = dino_channels
    new_config['dino_out_channels'] = dino_channels
    transformer = QuadStreamTransformer3DModel(**new_config)
    
    state_dict = {}
    index_file = os.path.join(transformer_path, "diffusion_pytorch_model.safetensors.index.json")
    single_file = os.path.join(transformer_path, "diffusion_pytorch_model.safetensors")

    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        shard_filenames = set(index_data["weight_map"].values())
        for shard_file in shard_filenames:
            shard_path = os.path.join(transformer_path, shard_file)
            shard_weights = load_file(shard_path)
            state_dict.update(shard_weights)
            del shard_weights 
    elif os.path.exists(single_file):
        state_dict = load_file(single_file)
    else:
        raise FileNotFoundError(f"Missing weights in {transformer_path}")

    keys_to_ignore = ["patch_embedding.weight", "patch_embedding.bias", "proj_out.weight", "proj_out.bias"]
    for key in keys_to_ignore:
        if key in state_dict:
            del state_dict[key]

    transformer.load_state_dict(state_dict, strict=False)
    del state_dict
    torch.cuda.empty_cache()
    transformer.to(dtype=dtype)
    return transformer

def generate_video(generator, model_helper, instance, generation_type):
    if generation_type == "i2v":
        image_path, prompt_list = model_helper.adapt(instance)
    elif generation_type == "t2v":
        prompt_list = model_helper.adapt(instance)
        image_path = None
    else:
        raise ValueError()

    all_generated_frames = []
    for i, prompt in enumerate(prompt_list):
        generated_frames = generator.generate_video(prompt=prompt, image_path=image_path)
        if generation_type == "i2v":
            image_path = model_helper.save_image(generated_frames[-1], image_path, i + 1)
        if i == 0:
            all_generated_frames += generated_frames
        else:
            all_generated_frames += generated_frames[1:]
    model_helper.save(all_generated_frames)

class GenWrapper:
    def __init__(self, pipe):
        self.pipe = pipe
        
    def generate_video(self, prompt, image_path=None):
        call_kwargs = {
            "prompt": prompt,
            "output_type": "np",
            "enable_inner_guidance": True
        }
        if image_path is not None:
            call_kwargs["image"] = image_path
            
        out = self.pipe(**call_kwargs)
        
        if isinstance(out, WanPipelineOutput): video = out.frames
        elif isinstance(out, tuple): video = out[0]
        else: video = out.frames

        v = video
        if hasattr(v, "ndim") and v.ndim == 5: v = v[0]
        
        frames = []
        if hasattr(v, "shape") and v.ndim == 4:
            if isinstance(v, torch.Tensor): v = v.cpu().numpy()
            
            for f_idx in range(v.shape[0]):
                frame = v[f_idx] 
                if frame.shape[-1] == 3:
                    pass
                elif frame.shape[0] == 3:
                    frame = np.transpose(frame, (1, 2, 0))
                
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).clip(0, 255)
                    frame = frame.astype(np.uint8)
                frame = np.ascontiguousarray(frame)
                pil_img = Image.fromarray(frame)
                frames.append(pil_img)                            
            return frames
        return [video]

def process_batch_local(
    model_helper: Any,
    data_batch: Sequence[Any],
    model_config: omegaconf.DictConfig,
    device: torch.device,
    pipeline_kwargs: Dict[str, Any],
):
    pretrained_path = pipeline_kwargs.get("pretrained_path")
    if not pretrained_path:
        raise ValueError("pretrained_path is missing")
        
    dtype = torch.float16
    
    try:
        custom_transformer = build_custom_transformer(
            pretrained_model_name_or_path=pretrained_path,
            dtype=dtype,
            dino_channels=16
        )
        base_pipe = WanPipeline.from_pretrained(
            pretrained_path,
            transformer=custom_transformer,
            torch_dtype=dtype,
            **(pipeline_kwargs.get("from_pretrained_kwargs", {}))
        )
        pipe = CustomWanPipeline(base_pipe)
    except Exception as e:
        logger.error("Pipeline init failed", exc_info=e)
        raise RuntimeError(f"Pipeline init failed: {e}")

    pipe.to(device)

    lora_path = pipeline_kwargs.get("lora_path", None)
    adapter_name = pipeline_kwargs.get("lora_adapter", "default")
    
    if lora_path:
        try:
            pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
        except Exception as e:
            logger.error("Failed to load custom weights", exc_info=e)

    generation_type = model_config.get("generation_type", "t2v")
    generator = GenWrapper(pipe)
    
    for instance in data_batch:
        try:
            generate_video(generator, model_helper, instance, generation_type)
        except Exception as err:
            logger.error("Instance processing failed", exc_info=err)

def run_worker_rank(rank, world_size, data_batches, model_helper, model_config, pipeline_kwargs):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")
    assigned = [batch for i, batch in enumerate(data_batches) if (i % world_size) == rank]
    for batch in assigned:
        process_batch_local(model_helper, batch, model_config, device, pipeline_kwargs)

def create_data_batches(data_instances, num_jobs):
    num_instances = len(data_instances)
    batch_size = max(len(data_instances) // num_jobs + 1, 5)
    return [data_instances[i : i + batch_size] for i in range(0, num_instances, batch_size)]

def main(
    model_name: str,
    pretrained_path: str,
    prompt_set: str = "",
    num_jobs: int = 1000,
    use_slurm: bool = False,  
    lora_path: Optional[str] = None,
    num_processes: Optional[int] = None,
    from_pretrained_kwargs: Optional[dict] = None,
    lora_adapter: str = "default",
    **unused_slurm_parameters,
):
    assert check_model(model_name), 'Model not exists!'
    model_type = get_model2type(type2model)[model_name]
    visual_movement_list = ["dynamic"]
    
    for visual_movement in visual_movement_list:
        data_instances, helper = GetHelpers(model_name, visual_movement, prompt_set)
        config_path = Path(__file__).parent / "configs" / f"{model_name}.yaml"
        model_config = omegaconf.OmegaConf.load(config_path)
        data_batches = create_data_batches(data_instances, num_jobs)
        
        if num_processes is None:
            num_processes = torch.cuda.device_count() if torch.cuda.is_available() else 1
        world_size = max(1, num_processes)
        
        pipeline_kwargs = {
            "pretrained_path": pretrained_path,
            "lora_path": lora_path,
            "lora_adapter": lora_adapter,
            "from_pretrained_kwargs": from_pretrained_kwargs or {},
        }
        
        if world_size == 1:
            run_worker_rank(0, world_size, data_batches, helper, model_config, pipeline_kwargs)
        else:
            mp.start_processes(
                run_worker_rank,
                args=(world_size, data_batches, helper, model_config, pipeline_kwargs),
                nprocs=world_size,
                join=True
            )

if __name__ == "__main__":
    fire.Fire(main)