import functools
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from torchvision.transforms import ToTensor
from .custome import CustomWanPipeline

import PIL.Image
import torch
import torch.nn as nn
import math 
import numpy as np
from accelerate import init_empty_weights
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanPipeline,
    WanTransformer3DModel as OriginalWanTransformer3DModel, 
)
from diffusers.configuration_utils import register_to_config

from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from transformers import AutoModel, AutoTokenizer, UMT5EncoderModel

import finetune.functional as FF
from finetune.data import VideoArtifact
from finetune.logging import get_logger
from finetune.models.modeling_utils import ModelSpecification
from finetune.processors import ProcessorMixin, T5Processor
from finetune.typing import ArtifactType, SchedulerType
from finetune.utils import get_non_null_items, safetensors_torch_save_function

logger = get_logger()

class DualStreamTransformer3DModel(OriginalWanTransformer3DModel):
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

class DinoFeatureProcessor(ProcessorMixin):
    def __init__(self, output_names: List[str], dino_feature_root: str):
        super().__init__()
        assert len(output_names) == 1, "DinoFeatureProcessor should have exactly one output name."
        self.output_names = output_names
        self.dino_feature_root = dino_feature_root

    def forward(
        self,
        vae: AutoencoderKLWan,
        video_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        device = vae.device
        dtype = vae.dtype

        video_path = video_path
        if not video_path:
            logger.warning("video_path not in kwargs. Skipping DINO features.")
            return {self.output_names[0]: None}
        video_identifier = os.path.splitext(os.path.basename(video_path[0]))[0]

        dino_feature_path = os.path.join(self.dino_feature_root, f"{video_identifier}.pth")

        if not os.path.exists(dino_feature_path):
            logger.warning(f"DINO feature file not found: {dino_feature_path}. Skipping.")
            return {self.output_names[0]: None}

        dino_data_dict = torch.load(dino_feature_path, map_location=device, weights_only=True)
        try:
            dino_latents = dino_data_dict['features'].to(dtype=dtype)
        except Exception as e:
            raise RuntimeError(f"Failed to load DINO feature file: {dino_feature_path}, shape is {dino_data_dict['features'].shape}, data is {dino_data_dict['features'][0, :, :5, :5]}") from e

        if dino_latents.ndim == 4:
            dino_latents = dino_latents.unsqueeze(0) # [1, C, T, H, W]

        return {self.output_names[0]: dino_latents}

class VGGTFeatureProcessor(ProcessorMixin):
    def __init__(self, output_names: List[str], vggt_feature_root: str):
        super().__init__()
        assert len(output_names) == 1, "VGGTFeatureProcessor should have exactly one output name."
        self.output_names = output_names
        self.vggt_feature_root = vggt_feature_root

    def forward(
        self,
        vae: AutoencoderKLWan,
        video_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        device = vae.device
        dtype = vae.dtype

        if not video_path:
            logger.warning("video_path not in kwargs. Skipping VGGT features.")
            return {self.output_names[0]: None}
        
        current_video_path = video_path[0] if isinstance(video_path, list) else video_path
        video_identifier = os.path.splitext(os.path.basename(current_video_path))[0]

        vggt_feature_path = os.path.join(self.vggt_feature_root, f"{video_identifier}.pth")

        if not os.path.exists(vggt_feature_path):
            logger.warning(f"VGGT feature file not found: {vggt_feature_path}. Skipping.")
            return {self.output_names[0]: None}

        try:
            vggt_data_dict = torch.load(vggt_feature_path, map_location=device, weights_only=True)
            vggt_latents = vggt_data_dict['features'].to(dtype=dtype) 
        except Exception as e:
            raise RuntimeError(f"Failed to load VGGT feature file: {vggt_feature_path}") from e
        
        feature_mean = vggt_latents.mean(dim=(1,2,3), keepdim=True)  
        feature_std  = vggt_latents.std(dim=(1,2,3), keepdim=True)   
        vggt_latents = (vggt_latents - feature_mean) / (feature_std + 1e-6)

        if vggt_latents.shape[1] == 81:
            first_frame = vggt_latents[:, :1, :, :]
            remaining_frames = vggt_latents[:, 1:, :, :]
            C, T_rest, H, W = remaining_frames.shape
            pooled_frames = remaining_frames.view(C, T_rest // 4, 4, H, W).mean(dim=2)
            vggt_latents = torch.cat([first_frame, pooled_frames], dim=1)

        if vggt_latents.ndim == 4:
            vggt_latents = vggt_latents.unsqueeze(0) # [1, C, T, H, W]

        return {self.output_names[0]: vggt_latents}

class FlowFeatureProcessor(ProcessorMixin):
    def __init__(self, output_names: List[str], flow_feature_root: str):
        super().__init__()
        assert len(output_names) == 1
        self.output_names = output_names
        self.flow_feature_root = flow_feature_root

    def forward(self, vae: AutoencoderKLWan, video_path: Optional[str] = None, **kwargs) -> Dict[str, torch.Tensor]:
        device = vae.device
        dtype = vae.dtype
        if not video_path:
            return {self.output_names[0]: None}
        
        v_path = video_path[0] if isinstance(video_path, list) else video_path
        video_identifier = os.path.splitext(os.path.basename(v_path))[0]
        flow_feature_path = os.path.join(self.flow_feature_root, f"{video_identifier}.pt")

        if not os.path.exists(flow_feature_path):
            return {self.output_names[0]: None}

        try:
            flow_latents = torch.load(flow_feature_path, map_location=device, weights_only=True)
            flow_latents = flow_latents.to(device=device, dtype=dtype)
        except Exception as e:
            raise RuntimeError(f"Error loading FLOW features: {e}")
        
        if flow_latents.ndim == 4:
            flow_latents = flow_latents.unsqueeze(0)
        
        # Normalize
        reduce_dims = tuple(range(1, flow_latents.ndim))
        feature_mean = flow_latents.mean(dim=reduce_dims, keepdim=True)
        feature_std = flow_latents.std(dim=reduce_dims, keepdim=True)
        flow_latents = (flow_latents - feature_mean) / (feature_std + 1e-6)

        return {self.output_names[0]: flow_latents}

class WanLatentEncodeProcessor(ProcessorMixin):
    r"""
    Processor to encode image/video into latents using the Wan VAE.
    """
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
        latents_mean = torch.tensor(vae.config.latents_mean)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std)
        return {self.output_names[0]: latents, self.output_names[1]: latents_mean, self.output_names[2]: latents_std}

class WanModelSpecification(ModelSpecification):
    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        tokenizer_id: Optional[str] = None,
        text_encoder_id: Optional[str] = None,
        transformer_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        text_encoder_dtype: torch.dtype = torch.bfloat16,
        transformer_dtype: torch.dtype = torch.bfloat16,
        vae_dtype: torch.dtype = torch.bfloat16,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        condition_model_processors: List[ProcessorMixin] = None,
        latent_model_processors: List[ProcessorMixin] = None,
        
        dino_feature_root: Optional[str] = None,
        vggt_feature_root: Optional[str] = None,
        flow_feature_root: Optional[str] = None,
        
        dino_in_channels: Optional[int] = None,
        dino_out_channels: Optional[int] = None,
        vggt_in_channels: Optional[int] = None, 
        vggt_out_channels: Optional[int] = None,
        flow_in_channels: Optional[int] = None, 
        flow_out_channels: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer_id=tokenizer_id,
            text_encoder_id=text_encoder_id,
            transformer_id=transformer_id,
            vae_id=vae_id,
            text_encoder_dtype=text_encoder_dtype,
            transformer_dtype=transformer_dtype,
            vae_dtype=vae_dtype,
            revision=revision,
            cache_dir=cache_dir,
        )
        
        self.dino_in_channels = dino_in_channels
        self.dino_out_channels = dino_out_channels
        self.vggt_in_channels = vggt_in_channels
        self.vggt_out_channels = vggt_out_channels
        self.flow_in_channels = flow_in_channels
        self.flow_out_channels = flow_out_channels

        if condition_model_processors is None:
            condition_model_processors = [T5Processor(["encoder_hidden_states", "__drop__"])]
        
        if latent_model_processors is None:
            latent_model_processors = [
                WanLatentEncodeProcessor(["latents", "latents_mean", "latents_std"]),
                DinoFeatureProcessor(
                    output_names=["dino_latents"],
                    dino_feature_root=dino_feature_root
                ),
                VGGTFeatureProcessor(
                    output_names=["vggt_latents"],
                    vggt_feature_root=vggt_feature_root
                ),
                FlowFeatureProcessor(
                    output_names=["flow_latents"],
                    flow_feature_root=flow_feature_root
                ),
            ]

        self.condition_model_processors = condition_model_processors
        self.latent_model_processors = latent_model_processors

    @property
    def _resolution_dim_keys(self):
        return {"latents": (2, 3, 4)}

    def load_condition_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}
        if self.tokenizer_id is not None:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id, **common_kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="tokenizer", **common_kwargs
            )
        if self.text_encoder_id is not None:
            text_encoder = AutoModel.from_pretrained(
                self.text_encoder_id, torch_dtype=self.text_encoder_dtype, **common_kwargs
            )
        else:
            text_encoder = UMT5EncoderModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="text_encoder",
                torch_dtype=self.text_encoder_dtype,
                **common_kwargs,
            )
        return {"tokenizer": tokenizer, "text_encoder": text_encoder}

    def load_latent_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}
        if self.vae_id is not None:
            vae = AutoencoderKLWan.from_pretrained(self.vae_id, torch_dtype=self.vae_dtype, **common_kwargs)
        else:
            vae = AutoencoderKLWan.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="vae", torch_dtype=self.vae_dtype, **common_kwargs
            )
        models = {"vae": vae}
        return models

    def load_diffusion_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        transformer_path_or_id = self.transformer_id or os.path.join(self.pretrained_model_name_or_path, "transformer")
        
        original_config = DualStreamTransformer3DModel.load_config(
            transformer_path_or_id, **common_kwargs
        )
        new_config = original_config.copy()

        new_config['dino_in_channels'] = self.dino_in_channels
        new_config['dino_out_channels'] = self.dino_out_channels
        new_config['vggt_in_channels'] = self.vggt_in_channels
        new_config['vggt_out_channels'] = self.vggt_out_channels
        new_config['flow_in_channels'] = self.flow_in_channels
        new_config['flow_out_channels'] = self.flow_out_channels

        transformer = DualStreamTransformer3DModel(**new_config)

        original_transformer = OriginalWanTransformer3DModel.from_pretrained(
            transformer_path_or_id, torch_dtype=self.transformer_dtype, **common_kwargs
        )
        original_sd = original_transformer.state_dict()
        mismatched_keys = ["patch_embedding.weight", "patch_embedding.bias", "proj_out.weight", "proj_out.bias"]
        copied_weights = {}
        
        for k in mismatched_keys:
            if k in original_sd:
                copied_weights[k] = original_sd.pop(k) 
            else:
                raise KeyError(f"Key {k} not found in original checkpoint.")

        transformer.load_state_dict(original_sd, strict=False)
        logger.info("Backbone weights loaded successfully.")

        with torch.no_grad():
            old_in_w = copied_weights["patch_embedding.weight"] 
            current_in_c = old_in_w.shape[1] 

            transformer.patch_embedding.weight.data[:, :current_in_c, ...] = old_in_w
            transformer.patch_embedding.weight.data[:, current_in_c:, ...] = 0.0
            
            transformer.patch_embedding.bias.data = copied_weights["patch_embedding.bias"]

            old_out_w = copied_weights["proj_out.weight"]
            old_out_b = copied_weights["proj_out.bias"] 
            
            patch_size = transformer.config.patch_size
            patch_vol = math.prod(patch_size)
            hidden_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim 
            
            old_ch = old_out_w.shape[0] // patch_vol 
            new_ch = transformer.config.out_channels 

            old_out_w_reshaped = old_out_w.view(patch_vol, old_ch, hidden_dim)          
            new_out_w_reshaped = torch.zeros(patch_vol, new_ch, hidden_dim, 
                                             dtype=old_out_w.dtype, device=old_out_w.device)
            
            new_out_w_reshaped[:, :old_ch, :] = old_out_w_reshaped
            transformer.proj_out.weight.data = new_out_w_reshaped.reshape(patch_vol * new_ch, hidden_dim)
            
            old_out_b_reshaped = old_out_b.view(patch_vol, old_ch)         
            new_out_b_reshaped = torch.zeros(patch_vol, new_ch, dtype=old_out_b.dtype, device=old_out_b.device)         
            new_out_b_reshaped[:, :old_ch] = old_out_b_reshaped
            transformer.proj_out.bias.data = new_out_b_reshaped.reshape(patch_vol * new_ch)
        
        logger.info("Weight copy and zero-initialization complete for 4 streams.")

        del original_transformer
        del original_sd
        del copied_weights
        
        transformer.to(self.transformer_dtype)

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="scheduler",
            shift=5.0 
        )

        return {"transformer": transformer, "scheduler": scheduler}

    def load_pipeline(
        self,
        tokenizer: Optional[AutoTokenizer] = None,
        text_encoder: Optional[UMT5EncoderModel] = None,
        transformer: Optional[DualStreamTransformer3DModel] = None,
        vae: Optional[AutoencoderKLWan] = None,
        scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None,
        enable_slicing: bool = False,
        enable_tiling: bool = False,
        enable_model_cpu_offload: bool = False,
        training: bool = False,
        **kwargs,
    ) -> WanPipeline:
        
        components = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
        }

        required_t2v = ["tokenizer", "text_encoder", "transformer", "vae", "scheduler"]
        missing_comps = [name for name in required_t2v if components.get(name) is None]
        if missing_comps:
            raise ValueError(f"WanPipeline missing required components: {missing_comps}")

        pipe = WanPipeline(
            tokenizer=components["tokenizer"],
            text_encoder=components["text_encoder"],
            transformer=components["transformer"],
            vae=components["vae"],
            scheduler=components["scheduler"],
        )

        pipe.text_encoder.to(self.text_encoder_dtype)
        pipe.vae.to(self.vae_dtype)
        if not training:
            pipe.transformer.to(self.transformer_dtype)
        if enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()

        return CustomWanPipeline(pipe)

    @torch.no_grad()
    def prepare_conditions(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        caption: str,
        max_sequence_length: int = 512,
        **kwargs,
    ) -> Dict[str, Any]:
        conditions = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "caption": caption,
            "max_sequence_length": max_sequence_length,
            **kwargs,
        }
        input_keys = set(conditions.keys())
        conditions = super().prepare_conditions(**conditions)
        conditions = {k: v for k, v in conditions.items() if k not in input_keys}
        return conditions

    @torch.no_grad()
    def prepare_latents(
        self,
        vae: AutoencoderKLWan,
        video: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        conditions = {
            "vae": vae,
            "video": video,
            "generator": generator,
            "compute_posterior": False,
            **kwargs,
        }
        input_keys = set(conditions.keys())
        processed_conditions = super().prepare_latents(**conditions)
        final_conditions = {k: v for k, v in processed_conditions.items() if k not in input_keys}
        return final_conditions
    
    def forward(
        self,
        transformer: DualStreamTransformer3DModel,
        condition_model_conditions: Dict[str, torch.Tensor],
        latent_model_conditions: Dict[str, torch.Tensor],
        sigmas: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        compute_posterior = False  
        
        latents = latent_model_conditions.pop("latents")
        latents_mean = latent_model_conditions.pop("latents_mean")
        latents_std = latent_model_conditions.pop("latents_std")

        mu, logvar = torch.chunk(latents, 2, dim=1)
        mu = self._normalize_latents(mu, latents_mean, latents_std)
        logvar = self._normalize_latents(logvar, latents_mean, latents_std)
        latents = torch.cat([mu, logvar], dim=1)

        posterior = DiagonalGaussianDistribution(latents)
        latents = posterior.sample(generator=generator) 
        del posterior

        dino_latents = latent_model_conditions.pop("dino_latents", None) 
        vggt_latents = latent_model_conditions.pop("vggt_latents", None)
        flow_latents = latent_model_conditions.pop("flow_latents", None)
        
        if dino_latents is None or vggt_latents is None or flow_latents is None:
             raise ValueError("Missing latents for DINO, VGGT or FLOW.")

        noise_vae = torch.zeros_like(latents).normal_(generator=generator)
        noise_dino = torch.zeros_like(dino_latents).normal_(generator=generator)
        noise_vggt = torch.zeros_like(vggt_latents).normal_(generator=generator)
        noise_flow = torch.zeros_like(flow_latents).normal_(generator=generator)
        
        vae_noisy_latents = FF.flow_match_xt(latents, noise_vae, sigmas)
        dino_noisy_latents = FF.flow_match_xt(dino_latents, noise_dino, sigmas)
        vggt_noisy_latents = FF.flow_match_xt(vggt_latents, noise_vggt, sigmas) 
        flow_noisy_latents = FF.flow_match_xt(flow_latents, noise_flow, sigmas) 
        
        timesteps = (sigmas.flatten() * 1000.0).long()

        noisy_latents = torch.cat([vae_noisy_latents, dino_noisy_latents, vggt_noisy_latents, flow_noisy_latents], dim=1)
            
        latent_model_conditions["hidden_states"] = noisy_latents.to(latents.dtype)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = transformer(
                **latent_model_conditions,
                **condition_model_conditions,
                timestep=timesteps,
                return_dict=False,
            )[0]
        
        vae_out_channels = transformer.config.original_out_channels
        dino_out_channels = transformer.config.dino_out_channels
        vggt_out_channels = transformer.config.vggt_out_channels
        flow_out_channels = transformer.config.flow_out_channels
        
        pred_vae, pred_dino, pred_vggt, pred_flow = torch.split(
            pred, 
            [vae_out_channels, dino_out_channels, vggt_out_channels, flow_out_channels], 
            dim=1
        )

        target_vae = FF.flow_match_target(noise_vae, latents)
        target_dino = FF.flow_match_target(noise_dino, dino_latents)
        target_vggt = FF.flow_match_target(noise_vggt, vggt_latents)
        target_flow = FF.flow_match_target(noise_flow, flow_latents)

        return pred_vae, target_vae, pred_dino, target_dino, pred_vggt, target_vggt, pred_flow, target_flow, sigmas

    def validation(
        self,
        pipeline: WanPipeline,
        prompt: str,
        video: Optional[List[PIL.Image.Image]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> List[ArtifactType]:
        generation_kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "return_dict": True,
            "output_type": "pil",
        }
        device=pipeline.device
        pipeline = pipeline.to(device)
        generation_kwargs = get_non_null_items(generation_kwargs)
        video = pipeline(**generation_kwargs).frames[0]
        return [VideoArtifact(value=video)]

    def _save_lora_weights(
        self,
        directory: str,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
        metadata: Optional[Dict[str, str]] = None,
        *args,
        **kwargs,
    ) -> None:
        pipeline_cls = WanPipeline
        
        if transformer_state_dict is None:
            logger.error("[DEBUG] transformer_state_dict is None when calling _save_lora_weights()!")
    
        if transformer_state_dict is not None:
            pipeline_cls.save_lora_weights(
                directory,
                transformer_lora_layers=transformer_state_dict,
                save_function=functools.partial(safetensors_torch_save_function, metadata=metadata),
                safe_serialization=True,
            )
        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))

    def _save_model(
        self,
        directory: str,
        transformer: DualStreamTransformer3DModel, 
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
    ) -> None:
        if transformer_state_dict is not None:
            with init_empty_weights():
                transformer_copy = DualStreamTransformer3DModel.from_config(transformer.config)
            transformer_copy.load_state_dict(transformer_state_dict, strict=True, assign=True)
            transformer_copy.save_pretrained(os.path.join(directory, "transformer"))
        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))

    @staticmethod
    def _normalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents