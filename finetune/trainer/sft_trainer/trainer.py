import functools
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
import cv2
import numpy as np
from PIL import Image

import datasets.distributed
from finetune.models.wan.custome import CustomWanPipeline
import torch
import torch.nn as nn
import wandb

from diffusers import DiffusionPipeline
from diffusers.hooks import apply_layerwise_casting
from diffusers.training_utils import cast_training_params
from diffusers.utils import export_to_video
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, get_peft_model_state_dict
from tqdm import tqdm

from finetune import data, logging, models, optimizer, parallel, utils
from finetune.args import BaseArgsType
from finetune.config import TrainingType
from finetune.state import TrainState

from ..base import Trainer
from .config import SFTFullRankConfig, SFTLowRankConfig
from torch.utils.data.dataloader import default_collate

os.environ["TOKENIZERS_PARALLELISM"] = "false"
ArgsType = Union[BaseArgsType, SFTFullRankConfig, SFTLowRankConfig]

logger = logging.get_logger()

cv2.setNumThreads(0) 
cv2.ocl.setUseOpenCL(False)

def save_video_opencv(frames: List[Image.Image], output_path: str, fps: int = 27):
    if len(frames) == 0:
        return
    frame = frames[0].convert("RGB")
    w, h = frame.size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for pil_img in frames:
        rgb_img = pil_img.convert("RGB")
        np_img = np.array(rgb_img)
        bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_img)
    video_writer.release()


class SFTTrainer(Trainer):
    def __init__(self, args: ArgsType, model_specification: models.ModelSpecification) -> None:
        super().__init__(args)

        self.tokenizer = None
        self.tokenizer_2 = None
        self.tokenizer_3 = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.text_encoder_3 = None
        self.transformer = None
        self.unet = None
        self.vae = None
        self.scheduler = None
        self.optimizer = None
        self.lr_scheduler = None
        self.checkpointer = None
        self.model_specification = model_specification
        self._are_condition_models_loaded = False

    def run(self) -> None:
        try:
            self._prepare_models()
            self.state.parallel_backend.wait_for_everyone()
            self._prepare_trainable_parameters()
            self.state.parallel_backend.wait_for_everyone()
            self._prepare_for_training()
            self._prepare_dataset()
            self._prepare_checkpointing()
            self._train()
        except Exception as e:
            logger.error(f"Error during training: {e}")
            self.state.parallel_backend.destroy()
            raise e

    def _prepare_models(self) -> None:
        logger.info("Initializing models")
        diffusion_components = self.model_specification.load_diffusion_models()
        self._set_components(diffusion_components)
        if self.state.parallel_backend.pipeline_parallel_enabled:
            raise NotImplementedError("Pipeline parallelism is not supported yet. This will be supported in the future.")

    def _prepare_trainable_parameters(self) -> None:
        logger.info("Initializing trainable parameters")
        parallel_backend = self.state.parallel_backend
        if self.args.training_type == TrainingType.FULL_FINETUNE:
            logger.info("Finetuning transformer with no additional parameters")
            utils.set_requires_grad([self.transformer], True)
        else:
            logger.info("Finetuning transformer with PEFT parameters")
            utils.set_requires_grad([self.transformer], False)

        if self.args.training_type == TrainingType.LORA and "transformer" in self.args.layerwise_upcasting_modules:
            apply_layerwise_casting(
                self.transformer,
                storage_dtype=self.args.layerwise_upcasting_storage_dtype,
                compute_dtype=self.args.transformer_dtype,
                skip_modules_pattern=self.args.layerwise_upcasting_skip_modules_pattern,
                non_blocking=True,
            )
        if self.args.training_type == TrainingType.LORA:
            modules_to_save = ["patch_embedding", "proj_out"]
            transformer_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.lora_alpha,
                init_lora_weights=True,
                target_modules=self.args.target_modules,
                modules_to_save=modules_to_save,
            )
            self.transformer.add_adapter(transformer_lora_config)

            for name, param in self.transformer.named_parameters():
                if any(m in name for m in modules_to_save):
                    param.requires_grad = True
                    logger.debug(f"Unfreezing parameter: {name}")

        if parallel_backend.data_sharding_enabled:
            self.transformer.to(dtype=self.args.transformer_dtype)
        else:
            if self.args.training_type == TrainingType.LORA:
                cast_training_params([self.transformer], dtype=torch.float32)
                target_dtype = self.args.transformer_dtype
                self.transformer.patch_embedding.to(dtype=target_dtype)
                self.transformer.proj_out.to(dtype=target_dtype)
                    
    def _prepare_for_training(self) -> None:
        parallel_backend = self.state.parallel_backend
        model_specification = self.model_specification
        if parallel_backend.context_parallel_enabled:
            parallel_backend.apply_context_parallel(self.transformer, parallel_backend.get_mesh()["cp"])
        if parallel_backend.tensor_parallel_enabled:
            model_specification.apply_tensor_parallel(
                backend=parallel.ParallelBackendEnum.PTD,
                device_mesh=parallel_backend.get_mesh()["tp"],
                transformer=self.transformer,
            )
        if self.args.gradient_checkpointing:
            utils.apply_activation_checkpointing(self.transformer, checkpointing_type="full")
        self._maybe_torch_compile()
        if parallel_backend.data_sharding_enabled:
            if self.args.parallel_backend == "accelerate":
                raise NotImplementedError("Data sharding is not supported with Accelerate yet.")
            dp_method = "HSDP" if parallel_backend.data_replication_enabled else "FSDP"
            logger.info(f"Applying {dp_method} on the model")
            if parallel_backend.data_replication_enabled or parallel_backend.context_parallel_enabled:
                dp_mesh_names = ("dp_replicate", "dp_shard_cp")
            else:
                dp_mesh_names = ("dp_shard_cp",)

            self.transformer = parallel_backend.apply_fsdp2(
                model=self.transformer,
                param_dtype=self.args.transformer_dtype,
                reduce_dtype=torch.float32,
                output_dtype=None,
                pp_enabled=parallel_backend.pipeline_parallel_enabled,
                cpu_offload=False,
                device_mesh=parallel_backend.get_mesh()[dp_mesh_names],
            )
        elif parallel_backend.data_replication_enabled:
            if parallel_backend.get_mesh().ndim > 1:
                raise ValueError("DDP not supported for > 1D parallelism")
            logger.info("Applying DDP to the model")
            self.transformer = parallel_backend.apply_ddp(self.transformer, parallel_backend.get_mesh())
        else:
            self.transformer = parallel_backend.prepare_model(self.transformer)

        self._move_components_to_device()
        model_parts = [self.transformer]
        self.state.num_trainable_parameters = sum(p.numel() for m in model_parts for p in m.parameters() if p.requires_grad)
        logger.info("Initializing optimizer and lr scheduler")
        self.state.train_state = TrainState()
        self.optimizer = optimizer.get_optimizer(
            parallel_backend=self.args.parallel_backend,
            name=self.args.optimizer,
            model_parts=model_parts,
            learning_rate=self.args.lr,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            fused=False,
        )
        self.lr_scheduler = optimizer.get_lr_scheduler(
            parallel_backend=self.args.parallel_backend,
            name=self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps,
            num_training_steps=self.args.train_steps,
        )
        self.optimizer, self.lr_scheduler = parallel_backend.prepare_optimizer(self.optimizer, self.lr_scheduler)
        self._init_logging()
        self._init_trackers()
        self._init_directories_and_repositories()

    def _prepare_dataset(self) -> None:
        logger.info("Initializing dataset and dataloader")
        with open(self.args.dataset_config, "r") as file:
            dataset_configs = json.load(file)["datasets"]
        logger.info(f"Training configured to use {len(dataset_configs)} datasets")
        datasets_list = []
        for config in dataset_configs:
            data_root = config.pop("data_root", None)
            dataset_file = config.pop("dataset_file", None)
            dataset_type = config.pop("dataset_type")
            caption_options = config.pop("caption_options", {})
            if data_root is not None and dataset_file is not None:
                raise ValueError("Both data_root and dataset_file cannot be provided in the same dataset config.")
            dataset_name_or_root = data_root or dataset_file
            dataset = data.initialize_dataset(
                dataset_name_or_root, dataset_type, streaming=True, infinite=True, _caption_options=caption_options
            )
            if not dataset._precomputable_once and self.args.precomputation_once:
                raise ValueError(f"Dataset {dataset_name_or_root} does not support precomputing all embeddings at once.")
            logger.info(f"Initialized dataset: {dataset_name_or_root}")
            dataset = self.state.parallel_backend.prepare_dataset(dataset)
            dataset = data.wrap_iterable_dataset_for_preprocessing(dataset, dataset_type, config)
            datasets_list.append(dataset)
        dataset = data.combine_datasets(datasets_list, buffer_size=self.args.dataset_shuffle_buffer_size, shuffle=True)
        dataloader = self.state.parallel_backend.prepare_dataloader(
            dataset, batch_size=1, num_workers=self.args.dataloader_num_workers, pin_memory=self.args.pin_memory,
        )
        self.dataset = dataset
        self.dataloader = dataloader

    def _prepare_checkpointing(self) -> None:
        parallel_backend = self.state.parallel_backend

        def save_model_hook(state_dict: Dict[str, Any]) -> None:
            state_dict = utils.get_unwrapped_model_state_dict(state_dict)
            if parallel_backend.is_main_process:
                if self.args.training_type == TrainingType.LORA:
                    state_dict = get_peft_model_state_dict(self.transformer, state_dict)
                    # fmt: off
                    metadata = {
                        "r": self.args.rank,
                        "lora_alpha": self.args.lora_alpha,
                        "init_lora_weights": True,
                        "target_modules": self.args.target_modules,
                    }
                    metadata = {"lora_config": json.dumps(metadata, indent=4)}
                    # fmt: on
                    self.model_specification._save_lora_weights(
                        os.path.join(self.args.output_dir, "lora_weights", f"{self.state.train_state.step:06d}"),
                        state_dict,
                        self.scheduler,
                        metadata,
                    )
                elif self.args.training_type == TrainingType.FULL_FINETUNE:
                    self.model_specification._save_model(
                        os.path.join(self.args.output_dir, "model_weights", f"{self.state.train_state.step:06d}"),
                        self.transformer,
                        state_dict,
                        self.scheduler,
                    )
            parallel_backend.wait_for_everyone()

        enable_state_checkpointing = self.args.checkpointing_steps > 0
        self.checkpointer = parallel_backend.get_checkpointer(
            dataloader=self.dataloader,
            model_parts=[self.transformer],
            optimizers=self.optimizer,
            schedulers=self.lr_scheduler,
            states={"train_state": self.state.train_state},
            checkpointing_steps=self.args.checkpointing_steps,
            checkpointing_limit=self.args.checkpointing_limit,
            output_dir=self.args.output_dir,
            enable=enable_state_checkpointing,
            _callback_fn=save_model_hook,
        )

        resume_from_checkpoint = self.args.resume_from_checkpoint
        if resume_from_checkpoint == "latest":
            resume_from_checkpoint = -1
        if resume_from_checkpoint is not None:
            self.checkpointer.load(resume_from_checkpoint)

    def _train(self) -> None:
        logger.info("Starting training")
        parallel_backend = self.state.parallel_backend
        train_state = self.state.train_state
        device = parallel_backend.device
        dtype = self.args.transformer_dtype
        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")
        global_batch_size = self.args.batch_size * parallel_backend._dp_degree
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "train steps": self.args.train_steps,
            "per-replica batch size": self.args.batch_size,
            "global batch size": global_batch_size,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        resume_step = 0

        self.transformer.train()
        logger.info("Initializing data iterator...")
        data_iterator = iter(self.dataloader)

        if resume_step > 0:
            logger.info(f"Manual Resume: Fast-forwarding training state to step {resume_step}...")

            train_state.step = resume_step
            train_state.observed_data_samples = resume_step * self.args.batch_size * parallel_backend._dp_degree

            logger.info("Aligning LR Scheduler state...")
            for s in range(1, resume_step + 1):
                if s % self.args.gradient_accumulation_steps == 0:
                    try:
                        self.lr_scheduler.step()
                    except Exception as e:
                        logger.warning(f"lr_scheduler.step() raised while aligning at step {s}: {e}")
            samples_to_skip = resume_step * self.args.batch_size
            logger.info(f"Dataset skipping first {samples_to_skip} samples per replica (Physically consuming iterator)...")
            
            skip_pbar = tqdm(
                range(samples_to_skip), 
                desc="Skipping data", 
                disable=not parallel_backend.is_local_main_process,
                unit="sample"
            )
            
            try:
                for _ in skip_pbar:
                    next(data_iterator)
            except StopIteration:
                logger.error("Dataset exhausted while skipping samples! The resume step is larger than the dataset size.")
                return
            
            logger.info("Data iterator successfully fast-forwarded.")

        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=train_state.step,
            desc="Training steps",
            disable=not parallel_backend.is_local_main_process,
        )

        generator = torch.Generator(device=device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        scheduler_sigmas = utils.get_scheduler_sigmas(self.scheduler)
        scheduler_sigmas = (
            scheduler_sigmas.to(device=device, dtype=torch.float32) if scheduler_sigmas is not None else None
        )
        scheduler_alphas = utils.get_scheduler_alphas(self.scheduler)
        scheduler_alphas = (
            scheduler_alphas.to(device=device, dtype=torch.float32) if scheduler_alphas is not None else None
        )

        compute_posterior = False if self.args.enable_precomputation else (not self.args.precomputation_once)
        preprocessor = data.initialize_preprocessor(
            rank=parallel_backend.rank,
            world_size=parallel_backend.world_size,
            num_items=self.args.precomputation_items if self.args.enable_precomputation else 1,
            processor_fn={
                "condition": self.model_specification.prepare_conditions,
                "latent": functools.partial(
                    self.model_specification.prepare_latents, compute_posterior=compute_posterior
                ),
            },
            save_dir=self.args.precomputation_dir,
            enable_precomputation=self.args.enable_precomputation,
            enable_reuse=self.args.precomputation_reuse,
        )

        condition_iterator: Iterable[Dict[str, Any]] = None
        latent_iterator: Iterable[Dict[str, Any]] = None
        sampler = data.ResolutionSampler(
            batch_size=self.args.batch_size, dim_keys=self.model_specification._resolution_dim_keys
        )
        requires_gradient_step = True
        accumulated_loss = 0.0

        while (
            train_state.step < self.args.train_steps and train_state.observed_data_samples < self.args.max_data_samples
        ):
            if preprocessor.requires_data:
                condition_iterator, latent_iterator = self._prepare_data(preprocessor, data_iterator)

            with self.tracker.timed("timing/batch_preparation"):
                try:
                    condition_item = next(condition_iterator)
                    latent_item = next(latent_iterator)
                    sampler.consume(condition_item, latent_item)
                except StopIteration:
                    if requires_gradient_step:
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        requires_gradient_step = False
                    logger.info("Data exhausted. Exiting training loop.")
                    break
                if sampler.is_ready:
                    condition_batch, latent_batch = sampler.get_batch()
                    condition_model_conditions = self.model_specification.collate_conditions(condition_batch)
                    latent_model_conditions = self.model_specification.collate_latents(latent_batch)
                else:
                    continue

            train_state.step += 1
            train_state.observed_data_samples += self.args.batch_size * parallel_backend._dp_degree

            logger.debug(f"Starting training step ({train_state.step}/{self.args.train_steps})")
            latent_model_conditions = utils.align_device_and_dtype(latent_model_conditions, device, dtype)
            condition_model_conditions = utils.align_device_and_dtype(condition_model_conditions, device, dtype)
            latent_model_conditions = utils.make_contiguous(latent_model_conditions)
            condition_model_conditions = utils.make_contiguous(condition_model_conditions)
            sigmas = utils.prepare_sigmas(
                scheduler=self.scheduler,
                sigmas=scheduler_sigmas,
                batch_size=self.args.batch_size,
                num_train_timesteps=self.scheduler.config.num_train_timesteps,
                flow_weighting_scheme=self.args.flow_weighting_scheme,
                flow_logit_mean=self.args.flow_logit_mean,
                flow_logit_std=self.args.flow_logit_std,
                flow_mode_scale=self.args.flow_mode_scale,
                device=device,
                generator=self.state.generator,
            )
            sigmas = utils.expand_tensor_dims(sigmas, latent_model_conditions["latents"].ndim)
            with self.attention_provider_ctx(training=True):
                with self.tracker.timed("timing/forward"):
                    pred_vae, target_vae, pred_dino, target_dino, pred_vggt, target_vggt, pred_flow, target_flow, sigmas = self.model_specification.forward(
                        transformer=self.transformer,
                        scheduler=self.scheduler,
                        condition_model_conditions=condition_model_conditions,
                        latent_model_conditions=latent_model_conditions,
                        sigmas=sigmas,
                        compute_posterior=compute_posterior,
                    )
                timesteps = (sigmas * 1000.0).long()
                weights = utils.prepare_loss_weights(
                    scheduler=self.scheduler,
                    alphas=scheduler_alphas[timesteps] if scheduler_alphas is not None else None,
                    sigmas=sigmas,
                    flow_weighting_scheme=self.args.flow_weighting_scheme,
                )
                weights_vae = utils.expand_tensor_dims(weights, pred_vae.ndim)
                weights_dino = utils.expand_tensor_dims(weights, pred_dino.ndim)
                weights_vggt = utils.expand_tensor_dims(weights, pred_vggt.ndim)
                weights_flow = utils.expand_tensor_dims(weights, pred_flow.ndim)
                with self.tracker.timed("timing/backward"):
                    loss_vae = (weights_vae.float() * (pred_vae.float() - target_vae.float()).pow(2)).mean()
                    loss_dino = (weights_dino.float() * (pred_dino.float() - target_dino.float()).pow(2)).mean()
                    loss_vggt = (weights_vggt.float() * (pred_vggt.float() - target_vggt.float()).pow(2)).mean()
                    loss_flow = (weights_flow.float() * (pred_flow.float() - target_flow.float()).pow(2)).mean()

                    current_step = train_state.step
                    total_steps = self.args.train_steps
                    base_weight = 0.2
                    cosine_factor = 0.5 * (1 + math.cos(math.pi * current_step / total_steps))
                    dynamic_weight = base_weight * cosine_factor

                    dino_weight = dynamic_weight
                    vggt_weight = dynamic_weight
                    flow_weight = dynamic_weight

                    total_loss = loss_vae + dino_weight * loss_dino + vggt_weight * loss_vggt + flow_weight * loss_flow
                    logger.debug(f"Step {train_state.step}: total_loss={total_loss.item()}")
                    logs = {
                        "train/loss_vae": loss_vae.detach().item(),
                        "train/loss_dino": loss_dino.detach().item(),
                        "train/loss_vggt": loss_vggt.detach().item(),
                        "train/loss_flow": loss_flow.detach().item(),
                        "train/total_loss": total_loss.detach().item(),
                    }
                    if self.args.gradient_accumulation_steps > 1:
                        total_loss = total_loss / self.args.gradient_accumulation_steps
                    
                    if self.args.gradient_checkpointing:
                        if hasattr(self.transformer, "patch_embedding"):
                            for param in self.transformer.patch_embedding.parameters():
                                if param.requires_grad:
                                    total_loss += param.sum() * 0.0                        
                        if hasattr(self.transformer, "proj_out"):
                            for param in self.transformer.proj_out.parameters():
                                if param.requires_grad:
                                    total_loss += param.sum() * 0.0

                    total_loss.backward()
                accumulated_loss += total_loss.detach().item()
                requires_gradient_step = True
            model_parts = [self.transformer]
            grad_norm = utils.torch._clip_grad_norm_while_handling_failing_dtensor_cases(
                [p for m in model_parts for p in m.parameters()],
                self.args.max_grad_norm,
                foreach=True,
                pp_mesh=parallel_backend.get_mesh()["pp"] if parallel_backend.pipeline_parallel_enabled else None,
            )
            if train_state.step % self.args.gradient_accumulation_steps == 0:
                with self.tracker.timed("timing/optimizer_step"):
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                if grad_norm is not None:
                    grad_norm = grad_norm if isinstance(grad_norm, float) else grad_norm.detach().item()
                if (
                    parallel_backend.data_replication_enabled
                    or parallel_backend.data_sharding_enabled
                    or parallel_backend.context_parallel_enabled
                ):
                    dp_cp_mesh = parallel_backend.get_mesh()["dp_cp"]
                    if grad_norm is not None:
                        grad_norm = parallel.dist_mean(torch.tensor([grad_norm], device=device), dp_cp_mesh)
                    global_avg_loss, global_max_loss = (
                        parallel.dist_mean(torch.tensor([accumulated_loss], device=device), dp_cp_mesh),
                        parallel.dist_max(torch.tensor([accumulated_loss], device=device), dp_cp_mesh),
                    )
                else:
                    global_avg_loss = global_max_loss = accumulated_loss
                logs["train/global_avg_loss"] = global_avg_loss
                logs["train/global_max_loss"] = global_max_loss
                if grad_norm is not None:
                    logs["train/grad_norm"] = grad_norm
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)
                accumulated_loss = 0.0
                requires_gradient_step = False
            progress_bar.update(1)
            progress_bar.set_postfix(logs)
            if train_state.step % self.args.logging_steps == 0:
                logs["train/observed_data_samples"] = train_state.observed_data_samples
                parallel_backend.log(logs, step=train_state.step)
                train_state.log_steps.append(train_state.step)
            with self.tracker.timed("timing/checkpoint"):
                self.checkpointer.save(
                    step=train_state.step, _device=device, _is_main_process=parallel_backend.is_main_process
                )
            if train_state.step % self.args.validation_steps == 0:
                self._validate(step=train_state.step, final_validation=False)
        self.checkpointer.save(
            train_state.step, force=True, _device=device, _is_main_process=parallel_backend.is_main_process
        )
        parallel_backend.wait_for_everyone()
        self._validate(step=train_state.step, final_validation=True)
        self._delete_components()
        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")
        if parallel_backend.is_main_process and self.args.push_to_hub:
            upload_folder(
                repo_id=self.state.repo_id,
                folder_path=self.args.output_dir,
                ignore_patterns=[f"{self.checkpointer._prefix}_*"],
            )
        parallel_backend.destroy()

    def _validate(self, step: int, final_validation: bool = False) -> None:
        if self.args.validation_dataset_file is None:
            return
        logger.info("Starting validation")
        parallel_backend = self.state.parallel_backend
        dataset = data.ValidationDataset(self.args.validation_dataset_file)
        if parallel_backend._dp_degree > 1:
            dp_mesh = parallel_backend.get_mesh()["dp"]
            dp_local_rank, dp_world_size = dp_mesh.get_local_rank(), dp_mesh.size()
            dataset._data = datasets.distributed.split_dataset_by_node(dataset._data, dp_local_rank, dp_world_size)
        else:
            dp_mesh = None
            dp_local_rank, dp_world_size = parallel_backend.local_rank, 1
        validation_dataloader = data.DPDataLoader(
            dp_local_rank,
            dataset,
            batch_size=1,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=lambda items: items,
        )
        data_iterator = iter(validation_dataloader)
        main_process_prompts_to_filenames = {}
        all_processes_artifacts = []
        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}")
        seed = self.args.seed if self.args.seed is not None else 0
        generator = torch.Generator(device=parallel_backend.device).manual_seed(seed)
        pipeline = self._init_pipeline(final_validation=final_validation)

        pipeline = CustomWanPipeline(pipeline)
        pipeline = pipeline.to(parallel_backend.device)
        if final_validation and self.args.training_type == TrainingType.LORA:
            lora_weights_path = os.path.join(self.args.output_dir, "lora_weights", f"{self.state.train_state.step:06d}")            
            pipeline.load_lora_weights(lora_weights_path)

        self.transformer.eval()
        while True:
            NUM_SAMPLES = 1
            validation_data = next(data_iterator, None)
            if validation_data is None:
                break
            PROMPT = validation_data["prompt"]
            if isinstance(PROMPT, list):
                PROMPT = PROMPT[0]

            for vindex in range(NUM_SAMPLES):
                current_seed = seed + vindex
                generator.manual_seed(current_seed)
                logger.info(f"Generating sample {vindex}/{NUM_SAMPLES} with seed {current_seed}")
                with self.attention_provider_ctx(training=False):
                    validation_artifacts = self.model_specification.validation(
                        pipeline=pipeline, generator=generator, **validation_data
                    )

                VIDEO = validation_data.get("video", None)
                EXPORT_FPS = validation_data.get("export_fps", 16)
                prompt_filename = utils.string_to_filename(PROMPT)[:25]
                artifacts = {
                    "input_video": data.VideoArtifact(value=VIDEO),
                }
                for i, validation_artifact in enumerate(validation_artifacts):
                    if validation_artifact.value is None:
                        continue
                    artifacts.update({f"artifact_{i}": validation_artifact})
                logger.info(f"Saving artifacts variation index: {vindex}")
                for index, (key, artifact) in enumerate(list(artifacts.items())):
                    assert isinstance(artifact, (data.ImageArtifact, data.VideoArtifact))
                    if artifact.value is None:
                        continue
                    time_, rank, ext = int(time.time()), parallel_backend.rank, artifact.file_extension
                    filename = "validation-" if not final_validation else "final-"
                    filename += f"{step}-{rank}-{index}-{prompt_filename}-{time_}.{ext}"
                    output_filename = os.path.join(self.args.output_dir, filename)
                    
                    if  ext in ["mp4", "jpg", "jpeg", "png"]:
                        main_process_prompts_to_filenames[PROMPT] = filename
                        if ext == "mp4":
                            save_video_opencv(artifact.value, output_filename, fps=EXPORT_FPS)
                        elif ext in ["jpg", "jpeg", "png"]:
                            artifact.value.save(output_filename)            
        parallel_backend.wait_for_everyone()
        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")
        pipeline.remove_all_hooks()
        del pipeline
        module_names = ["text_encoder", "text_encoder_2", "text_encoder_3", "vae"]
        if self.args.enable_precomputation:
            self._delete_components(module_names)
        torch.cuda.reset_peak_memory_stats(parallel_backend.device)
        all_artifacts = [None] * dp_world_size
        if dp_world_size > 1:
            torch.distributed.all_gather_object(all_artifacts, all_processes_artifacts)
        else:
            all_artifacts = [all_processes_artifacts]
        all_artifacts = [artifact for artifacts in all_artifacts for artifact in artifacts]
        if parallel_backend.is_main_process:
            tracker_key = "final" if final_validation else "validation"
            artifact_log_dict = {}
            image_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Image)]
            if len(image_artifacts) > 0:
                artifact_log_dict["images"] = image_artifacts
            video_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Video)]
            if len(video_artifacts) > 0:
                artifact_log_dict["videos"] = video_artifacts
            parallel_backend.log({tracker_key: artifact_log_dict}, step=step)
            if self.args.push_to_hub and final_validation:
                video_filenames = list(main_process_prompts_to_filenames.values())
                prompts = list(main_process_prompts_to_filenames.keys())
                utils.save_model_card(
                    args=self.args, repo_id=self.state.repo_id, videos=video_filenames, validation_prompts=prompts
                )
        parallel_backend.wait_for_everyone()
        if not final_validation:
            self._move_components_to_device()
            self.transformer.train()

    def _evaluate(self) -> None:
        raise NotImplementedError("Evaluation has not been implemented yet.")

    def _init_directories_and_repositories(self) -> None:
        if self.state.parallel_backend.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)
            self.state.output_dir = Path(self.args.output_dir)
            if self.args.push_to_hub:
                repo_id = self.args.hub_model_id or Path(self.args.output_dir).name
                self.state.repo_id = create_repo(token=self.args.hub_token, repo_id=repo_id, exist_ok=True).repo_id

    def _move_components_to_device(
        self, components: Optional[List[torch.nn.Module]] = None, device: Optional[Union[str, torch.device]] = None
    ) -> None:
        if device is None:
            device = self.state.parallel_backend.device
        if components is None:
            components = [
                self.text_encoder, self.text_encoder_2, self.text_encoder_3,
                self.transformer, self.vae,
            ]
        components = utils.get_non_null_items(components)
        components = list(filter(lambda x: hasattr(x, "to"), components))
        for component in components:
            component.to(device)

    def _set_components(self, components: Dict[str, Any]) -> None:
        all_component_names = self._all_component_names
        for component_name in all_component_names:
            existing_component = getattr(self, component_name, None)
            new_component = components.get(component_name, existing_component)
            setattr(self, component_name, new_component)

    def _delete_components(self, component_names: Optional[List[str]] = None) -> None:
        if component_names is None:
            component_names = self._all_component_names
        for component_name in component_names:
            setattr(self, component_name, None)
        utils.free_memory()
        utils.synchronize_device()

    def _init_pipeline(self, final_validation: bool = False) -> DiffusionPipeline:
            module_names = ["text_encoder", "text_encoder_2", "text_encoder_3", "transformer", "vae"]
            transformer_to_use = self.transformer
            if hasattr(self.transformer, "module"):
                transformer_to_use = self.transformer.module

            if not final_validation:
                logger.info("Initializing pipeline for intermediate validation (training=True).")
                module_names.remove("transformer")
                components_to_pass = {
                    "tokenizer": getattr(self, "tokenizer", None),
                    "tokenizer_2": getattr(self, "tokenizer_2", None),
                    "tokenizer_3": getattr(self, "tokenizer_3", None),
                    "text_encoder": getattr(self, "text_encoder", None),
                    "text_encoder_2": getattr(self, "text_encoder_2", None),
                    "text_encoder_3": getattr(self, "text_encoder_3", None),
                    "transformer": getattr(self, "transformer", None),
                    "vae": getattr(self, "vae", None),
                    "scheduler": getattr(self, "scheduler", None), 
                    "training": True,
                }
                
                pipeline = self.model_specification.load_pipeline(
                    **components_to_pass,
                    enable_slicing=self.args.enable_slicing,
                    enable_tiling=self.args.enable_tiling,
                    enable_model_cpu_offload=self.args.enable_model_cpu_offload,
                )
            else:

                self._delete_components()

                condition_models = self.model_specification.load_condition_models()
                latent_models = self.model_specification.load_latent_models()
                diffusion_models = self.model_specification.load_diffusion_models()

                components_to_pass = {
                    "tokenizer": condition_models.get("tokenizer"),
                    "text_encoder": condition_models.get("text_encoder"),
                    "vae": latent_models.get("vae"),
                    "transformer": diffusion_models["transformer"], 
                    "scheduler": diffusion_models["scheduler"],
                    "training": False,
                }

                pipeline = self.model_specification.load_pipeline(
                    **components_to_pass,
                    enable_slicing=self.args.enable_slicing,
                    enable_tiling=self.args.enable_tiling,
                    enable_model_cpu_offload=self.args.enable_model_cpu_offload,
                )
                  
            components = {module_name: getattr(pipeline, module_name, None) for module_name in module_names}
            self._set_components(components)
            if not self.args.enable_model_cpu_offload:
                self._move_components_to_device(list(components.values()))
            self._maybe_torch_compile()
            return pipeline

    def _prepare_data(
        self,
        preprocessor: Union[data.InMemoryDistributedDataPreprocessor, data.PrecomputedDistributedDataPreprocessor],
        data_iterator,
    ):
        if not self.args.enable_precomputation:
            if not self._are_condition_models_loaded:
                logger.info("Precomputation disabled. Loading in-memory data loaders. All components will be loaded on GPUs.")
                condition_components = self.model_specification.load_condition_models()
                latent_components = self.model_specification.load_latent_models()
                all_components = {**condition_components, **latent_components}
                self._set_components(all_components)
                self._move_components_to_device(list(all_components.values()))
                utils._enable_vae_memory_optimizations(self.vae, self.args.enable_slicing, self.args.enable_tiling)
                self._maybe_torch_compile()
            else:
                condition_components = {k: v for k in self._condition_component_names if (v := getattr(self, k, None))}
                latent_components = {k: v for k in self._latent_component_names if (v := getattr(self, k, None))}
            condition_iterator = preprocessor.consume(
                "condition", components=condition_components, data_iterator=data_iterator, generator=self.state.generator, cache_samples=True
            )
            latent_iterator = preprocessor.consume(
                "latent", components=latent_components, data_iterator=data_iterator, generator=self.state.generator, use_cached_samples=True, drop_samples=True
            )
            self._are_condition_models_loaded = True
        else:
            logger.info("Precomputed condition & latent data exhausted. Loading & preprocessing new data.")
            parallel_backend = self.state.parallel_backend
            if parallel_backend.world_size == 1:
                self._move_components_to_device([self.transformer], "cpu")
                utils.free_memory()
                utils.synchronize_device()
                torch.cuda.reset_peak_memory_stats(parallel_backend.device)
            consume_fn = preprocessor.consume_once if self.args.precomputation_once else preprocessor.consume
            condition_components, component_names, component_modules = {}, [], []
            if not self.args.precomputation_reuse:
                condition_components = self.model_specification.load_condition_models()
                component_names = list(condition_components.keys())
                component_modules = list(condition_components.values())
                self._set_components(condition_components)
                self._move_components_to_device(component_modules)
                self._maybe_torch_compile()
            condition_iterator = consume_fn(
                "condition", components=condition_components, data_iterator=data_iterator, generator=self.state.generator, cache_samples=True
            )
            self._delete_components(component_names)
            del condition_components, component_names, component_modules
            latent_components, component_names, component_modules = {}, [], []
            if not self.args.precomputation_reuse:
                latent_components = self.model_specification.load_latent_models()
                utils._enable_vae_memory_optimizations(self.vae, self.args.enable_slicing, self.args.enable_tiling)
                component_names = list(latent_components.keys())
                component_modules = list(latent_components.values())
                self._set_components(latent_components)
                self._move_components_to_device(component_modules)
                self._maybe_torch_compile()
            latent_iterator = consume_fn(
                "latent", components=latent_components, data_iterator=data_iterator, generator=self.state.generator, use_cached_samples=True, drop_samples=True
            )
            self._delete_components(component_names)
            del latent_components, component_names, component_modules
            if parallel_backend.world_size == 1:
                self._move_components_to_device([self.transformer])
        return condition_iterator, latent_iterator

    def _maybe_torch_compile(self):
        for model_name, compile_scope in zip(self.args.compile_modules, self.args.compile_scopes):
            model = getattr(self, model_name, None)
            if model is not None:
                logger.info(f"Applying torch.compile to '{model_name}' with scope '{compile_scope}'.")
                compiled_model = utils.apply_compile(model, compile_scope)
                setattr(self, model_name, compiled_model)

    def _get_training_info(self) -> Dict[str, Any]:
        info = self.args.to_dict()
        diffusion_args = info.get("diffusion_arguments", {})
        scheduler_name = self.scheduler.__class__.__name__ if self.scheduler is not None else ""
        if scheduler_name != "FlowMatchEulerDiscreteScheduler":
            filtered_diffusion_args = {k: v for k, v in diffusion_args.items() if "flow" not in k}
        else:
            filtered_diffusion_args = diffusion_args
        info.update({"diffusion_arguments": filtered_diffusion_args})
        return info

    _all_component_names = ["tokenizer", "tokenizer_2", "tokenizer_3", "text_encoder", "text_encoder_2", "text_encoder_3", "transformer", "unet", "vae", "scheduler"]
    _condition_component_names = ["tokenizer", "tokenizer_2", "tokenizer_3", "text_encoder", "text_encoder_2", "text_encoder_3"]
    _latent_component_names = ["vae"]
    _diffusion_component_names = ["transformer", "unet", "scheduler"]