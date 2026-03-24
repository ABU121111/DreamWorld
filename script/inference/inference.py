import argparse
import json
import os
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import datasets.distributed
import torch
import wandb
from diffusers.hooks import HookRegistry, ModelHook
from diffusers.utils import export_to_video

from finetune import data, get_logger, logging, parallel, patches, utils
from finetune.args import AttentionProviderInference
from finetune.config import ModelType
from finetune.models import ModelSpecification, attention_provider
from finetune.models.wan import WanModelSpecification
from finetune.parallel import ParallelBackendEnum
from finetune.state import ParallelBackendType
from finetune.utils import ArgsConfigMixin

logger = get_logger()


def main():
    try:
        import multiprocessing
        multiprocessing.set_start_method("fork")
    except Exception as e:
        logger.error(
            f'Failed to set multiprocessing start method to "fork". This can lead to poor performance, high memory usage, or crashes. '
            f"See: https://pytorch.org/docs/stable/notes/multiprocessing.html\n"
            f"Error: {e}"
        )

    try:
        args = BaseArgs()
        args = args.parse_args()

        model_specification_cls = get_model_specifiction_cls(args.model_name, args.inference_type)
        model_specification = model_specification_cls(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            tokenizer_id=args.tokenizer_id,
            tokenizer_2_id=args.tokenizer_2_id,
            tokenizer_3_id=args.tokenizer_3_id,
            text_encoder_id=args.text_encoder_id,
            text_encoder_2_id=args.text_encoder_2_id,
            text_encoder_3_id=args.text_encoder_3_id,
            transformer_id=args.transformer_id,
            vae_id=args.vae_id,
            text_encoder_dtype=args.text_encoder_dtype,
            text_encoder_2_dtype=args.text_encoder_2_dtype,
            text_encoder_3_dtype=args.text_encoder_3_dtype,
            transformer_dtype=args.transformer_dtype,
            vae_dtype=args.vae_dtype,
            revision=args.revision,
            cache_dir=args.cache_dir,
            
            dino_feature_root=args.dino_feature_root,
            vggt_feature_root=args.vggt_feature_root,
            flow_feature_root=args.flow_feature_root,
            dino_in_channels=args.dino_in_channels,
            dino_out_channels=args.dino_out_channels,
            vggt_in_channels=args.vggt_in_channels,
            vggt_out_channels=args.vggt_out_channels,
            flow_in_channels=args.flow_in_channels,
            flow_out_channels=args.flow_out_channels,
        )

        inferencer = Inference(args, model_specification)
        inferencer.run()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Exiting...")
    except Exception as e:
        logger.error(f"An error occurred during inference: {e}")
        logger.error(traceback.format_exc())


class InferenceType(str, Enum):
    TEXT_TO_IMAGE = "text_to_image"
    TEXT_TO_VIDEO = "text_to_video"


BaseArgsType = Union[
    "BaseArgs", "ParallelArgs", "ModelArgs", "InferenceArgs", "AttentionProviderArgs", "TorchConfigArgs"
]

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}

SUPPORTED_MODEL_CONFIGS = {
    ModelType.WAN: {
        InferenceType.TEXT_TO_VIDEO: WanModelSpecification,
    },
}


def get_model_specifiction_cls(model_name: str, inference_type: InferenceType) -> ModelSpecification:
    if model_name not in SUPPORTED_MODEL_CONFIGS:
        raise ValueError(
            f"Model {model_name} not supported. Supported models are: {list(SUPPORTED_MODEL_CONFIGS.keys())}"
        )
    if inference_type not in SUPPORTED_MODEL_CONFIGS[model_name]:
        raise ValueError(
            f"Inference type {inference_type} not supported for model {model_name}. Supported inference types are: {list(SUPPORTED_MODEL_CONFIGS[model_name].keys())}"
        )
    return SUPPORTED_MODEL_CONFIGS[model_name][inference_type]


@dataclass
class State:
    parallel_backend: ParallelBackendType = None
    generator: torch.Generator = None


class Inference:
    def __init__(self, args: BaseArgsType, model_specification: ModelSpecification):
        self.args = args
        self.model_specification = model_specification
        self.state = State()

        self.pipeline = None
        self.dataset = None
        self.dataloader = None

        self._init_distributed()
        self._init_config_options()

        patches.perform_patches_for_inference(args, self.state.parallel_backend)

    def run(self) -> None:
        try:
            self._prepare_pipeline()
            self._prepare_distributed()
            self._prepare_dataset()
            self._inference()
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            self.state.parallel_backend.destroy()
            raise e

    def _prepare_pipeline(self) -> None:
        logger.info("Initializing pipeline")

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

        self.pipeline = self.model_specification.load_pipeline(
            **components_to_pass,
            enable_slicing=self.args.enable_slicing,
            enable_tiling=self.args.enable_tiling,
            enable_model_cpu_offload=False,
        )

        if self.args.lora_path is not None:
            logger.info(f"Loading custom weights/LoRA from {self.args.lora_path}")
            self.pipeline.load_lora_weights(self.args.lora_path)

    def _prepare_distributed(self) -> None:
        parallel_backend = self.state.parallel_backend
        cp_mesh = parallel_backend.get_mesh("cp") if parallel_backend.context_parallel_enabled else None

        if parallel_backend.context_parallel_enabled:
            cp_mesh = parallel_backend.get_mesh()["cp"]
            parallel_backend.apply_context_parallel(self.pipeline.transformer, cp_mesh)

        registry = HookRegistry.check_if_exists_or_initialize(self.pipeline.transformer)
        hook = AttentionProviderHook(
            self.args.attn_provider, cp_mesh, self.args.cp_rotate_method, self.args.cp_reduce_precision
        )
        registry.register_hook(hook, "attn_provider")

        self._maybe_torch_compile()

        self._init_logging()
        self._init_trackers()
        self._init_directories()

    def _prepare_dataset(self) -> None:
        logger.info("Preparing dataset for inference")
        parallel_backend = self.state.parallel_backend

        dp_mesh = None
        if parallel_backend.data_replication_enabled:
            dp_mesh = parallel_backend.get_mesh("dp_replicate")
        if dp_mesh is not None:
            local_rank, dp_world_size = dp_mesh.get_local_rank(), dp_mesh.size()
        else:
            local_rank, dp_world_size = 0, 1

        dataset = data.ValidationDataset(self.args.dataset_file)
        dataset._data = datasets.distributed.split_dataset_by_node(dataset._data, local_rank, dp_world_size)
        dataloader = data.DPDataLoader(
            local_rank,
            dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=lambda items: items,
        )

        self.dataset = dataset
        self.dataloader = dataloader

    def _inference(self) -> None:
        parallel_backend = self.state.parallel_backend
        seed = self.args.seed if self.args.seed is not None else 0
        generator = torch.Generator(device=parallel_backend.device).manual_seed(seed)

        if parallel_backend._dp_degree > 1:
            dp_mesh = parallel_backend.get_mesh("dp")
            dp_local_rank, dp_world_size = dp_mesh.get_local_rank(), dp_mesh.size()
        else:
            dp_mesh = None
            dp_local_rank, dp_world_size = parallel_backend.local_rank, 1

        self.pipeline.to(parallel_backend.device)

        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory before inference start: {json.dumps(memory_statistics, indent=4)}")

        data_iterator = iter(self.dataloader)
        main_process_prompts_to_filenames = {}
        all_processes_artifacts = []

        while True:
            inference_data = next(data_iterator, None)
            if inference_data is None:
                break

            if isinstance(inference_data, list):
                inference_data = inference_data[0]
            with torch.inference_mode():
                inference_artifacts = self.model_specification.validation(
                    pipeline=self.pipeline, generator=generator, **inference_data
                )

            if dp_local_rank != 0:
                continue

            PROMPT = inference_data["prompt"]
            if isinstance(PROMPT, list):
                PROMPT = PROMPT[0]
            IMAGE = inference_data.get("image", None)
            VIDEO = inference_data.get("video", None)
            EXPORT_FPS = inference_data.get("export_fps", 30)

            prompt_filename = utils.string_to_filename(PROMPT)[:25]
            artifacts = {
                "input_image": data.ImageArtifact(value=IMAGE),
                "input_video": data.VideoArtifact(value=VIDEO),
            }

            for i, inference_artifact in enumerate(inference_artifacts):
                if inference_artifact.value is None:
                    continue
                artifacts.update({f"artifact_{i}": inference_artifact})

            for index, (key, artifact) in enumerate(list(artifacts.items())):
                assert isinstance(artifact, (data.ImageArtifact, data.VideoArtifact))
                if artifact.value is None:
                    continue

                time_, rank, ext = int(time.time()), parallel_backend.rank, artifact.file_extension
                filename = f"inference-{rank}-{index}-{prompt_filename}-{time_}.{ext}"
                output_filename = os.path.join(self.args.output_dir, filename)

                if parallel_backend.is_main_process and ext in ["mp4", "jpg", "jpeg", "png"]:
                    main_process_prompts_to_filenames[PROMPT] = filename

                if isinstance(artifact, data.ImageArtifact):
                    artifact.value.save(output_filename)
                    all_processes_artifacts.append(wandb.Image(output_filename, caption=PROMPT))
                elif isinstance(artifact, data.VideoArtifact):
                    export_to_video(artifact.value, output_filename, fps=EXPORT_FPS)
                    all_processes_artifacts.append(wandb.Video(output_filename, caption=PROMPT))

        parallel_backend.wait_for_everyone()
        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory after inference end: {json.dumps(memory_statistics, indent=4)}")

        all_artifacts = [None] * dp_world_size
        if dp_world_size > 1:
            torch.distributed.all_gather_object(all_artifacts, all_processes_artifacts)
        else:
            all_artifacts = [all_processes_artifacts]
        all_artifacts = [artifact for artifacts in all_artifacts for artifact in artifacts]

        if parallel_backend.is_main_process:
            tracker_key = "inference"
            artifact_log_dict = {}

            image_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Image)]
            if len(image_artifacts) > 0:
                artifact_log_dict["images"] = image_artifacts
            video_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Video)]
            if len(video_artifacts) > 0:
                artifact_log_dict["videos"] = video_artifacts
            parallel_backend.log({tracker_key: artifact_log_dict}, step=0)

        parallel_backend.wait_for_everyone()

    def _init_distributed(self) -> None:
        world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
        backend_cls: parallel.ParallelBackendType = parallel.get_parallel_backend_cls(self.args.parallel_backend)
        self.state.parallel_backend = backend_cls(
            world_size=world_size,
            pp_degree=self.args.pp_degree,
            dp_degree=self.args.dp_degree,
            dp_shards=self.args.dp_shards,
            cp_degree=self.args.cp_degree,
            tp_degree=self.args.tp_degree,
            backend="nccl",
            timeout=self.args.init_timeout,
            logging_dir=self.args.logging_dir,
            output_dir=self.args.output_dir,
        )
        if self.args.seed is not None:
            self.state.parallel_backend.enable_determinism(self.args.seed)

    def _init_logging(self) -> None:
        logging._set_parallel_backend(self.state.parallel_backend)
        logging.set_dependency_log_level(self.args.verbose, self.state.parallel_backend.is_local_main_process)
        logger.info("Initialized Finetrainers")

    def _init_trackers(self) -> None:
        trackers = [self.args.report_to]
        experiment_name = self.args.tracker_name or "finetune-inference"
        self.state.parallel_backend.initialize_trackers(
            trackers, experiment_name=experiment_name, config=self.args.to_dict(), log_dir=self.args.logging_dir
        )

    def _init_directories(self) -> None:
        if self.state.parallel_backend.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)

    def _init_config_options(self) -> None:
        if self.args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision(self.args.float32_matmul_precision)

    def _maybe_torch_compile(self):
        for model_name, compile_scope in zip(self.args.compile_modules, self.args.compile_scopes):
            model = getattr(self.pipeline, model_name, None)
            if model is not None:
                logger.info(f"Applying torch.compile to '{model_name}' with scope '{compile_scope}'.")
                compiled_model = utils.apply_compile(model, compile_scope)
                setattr(self.pipeline, model_name, compiled_model)


class AttentionProviderHook(ModelHook):
    def __init__(
            self,
            provider: str,
            mesh: Optional[torch.distributed.device_mesh.DeviceMesh] = None,
            rotate_method: str = "allgather",
            reduce_precision: bool = False,
    ):
        super().__init__()
        self.provider = provider
        self.mesh = mesh
        self.rotate_method = rotate_method
        self.convert_to_fp32 = not reduce_precision

    def new_forward(self, module: torch.nn.Module, *args, **kwargs) -> Any:
        with attention_provider(
                self.provider, mesh=self.mesh, convert_to_fp32=self.convert_to_fp32, rotate_method=self.rotate_method
        ):
            return self.fn_ref.original_forward(*args, **kwargs)


class ParallelArgs(ArgsConfigMixin):
    parallel_backend: ParallelBackendEnum = ParallelBackendEnum.ACCELERATE
    pp_degree: int = 1
    dp_degree: int = 1
    dp_shards: int = 1
    cp_degree: int = 1
    tp_degree: int = 1
    cp_rotate_method: str = "allgather"
    cp_reduce_precision: bool = False

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--parallel_backend", type=str, default="accelerate", choices=["accelerate", "ptd"])
        parser.add_argument("--pp_degree", type=int, default=1)
        parser.add_argument("--dp_degree", type=int, default=1)
        parser.add_argument("--dp_shards", type=int, default=1)
        parser.add_argument("--cp_degree", type=int, default=1)
        parser.add_argument("--tp_degree", type=int, default=1)
        parser.add_argument("--cp_rotate_method", type=str, default="allgather", choices=["allgather", "alltoall"])
        parser.add_argument("--cp_reduce_precision", action="store_true")

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        mapped_args.parallel_backend = argparse_args.parallel_backend
        mapped_args.pp_degree = argparse_args.pp_degree
        mapped_args.dp_degree = argparse_args.dp_degree
        mapped_args.dp_shards = argparse_args.dp_shards
        mapped_args.cp_degree = argparse_args.cp_degree
        mapped_args.tp_degree = argparse_args.tp_degree
        mapped_args.cp_rotate_method = argparse_args.cp_rotate_method
        mapped_args.cp_reduce_precision = argparse_args.cp_reduce_precision

    def validate_args(self, args: "BaseArgs"):
        if args.parallel_backend != "ptd":
            raise ValueError("Only 'ptd' parallel backend is supported for now.")
        if any(x > 1 for x in [args.pp_degree, args.dp_degree, args.dp_shards, args.tp_degree]):
            raise ValueError("Parallel degrees must be 1 except for `cp_degree` for now.")


class ModelArgs(ArgsConfigMixin):
    model_name: str = None
    pretrained_model_name_or_path: str = None
    revision: Optional[str] = None
    variant: Optional[str] = None
    cache_dir: Optional[str] = None
    tokenizer_id: Optional[str] = None
    tokenizer_2_id: Optional[str] = None
    tokenizer_3_id: Optional[str] = None
    text_encoder_id: Optional[str] = None
    text_encoder_2_id: Optional[str] = None
    text_encoder_3_id: Optional[str] = None
    transformer_id: Optional[str] = None
    vae_id: Optional[str] = None
    text_encoder_dtype: torch.dtype = torch.bfloat16
    text_encoder_2_dtype: torch.dtype = torch.bfloat16
    text_encoder_3_dtype: torch.dtype = torch.bfloat16
    transformer_dtype: torch.dtype = torch.bfloat16
    vae_dtype: torch.dtype = torch.bfloat16
    layerwise_upcasting_modules: List[str] = []
    layerwise_upcasting_storage_dtype: torch.dtype = torch.float8_e4m3fn
    layerwise_upcasting_skip_modules_pattern: List[str] = ["patch_embed", "pos_embed", "x_embedder", "context_embedder",
                                                           "time_embed", "^proj_in$", "^proj_out$", "norm"]
    enable_slicing: bool = False
    enable_tiling: bool = False

    dino_feature_root: Optional[str] = None
    vggt_feature_root: Optional[str] = None
    flow_feature_root: Optional[str] = None
    dino_in_channels: Optional[int] = None
    dino_out_channels: Optional[int] = None
    vggt_in_channels: Optional[int] = None
    vggt_out_channels: Optional[int] = None
    flow_in_channels: Optional[int] = None
    flow_out_channels: Optional[int] = None

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--model_name", type=str, required=True,
                            choices=[x.value for x in ModelType.__members__.values()])
        parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
        parser.add_argument("--revision", type=str, default=None, required=False)
        parser.add_argument("--variant", type=str, default=None)
        parser.add_argument("--cache_dir", type=str, default=None)
        parser.add_argument("--tokenizer_id", type=str, default=None)
        parser.add_argument("--tokenizer_2_id", type=str, default=None)
        parser.add_argument("--tokenizer_3_id", type=str, default=None)
        parser.add_argument("--text_encoder_id", type=str, default=None)
        parser.add_argument("--text_encoder_2_id", type=str, default=None)
        parser.add_argument("--text_encoder_3_id", type=str, default=None)
        parser.add_argument("--transformer_id", type=str, default=None)
        parser.add_argument("--vae_id", type=str, default=None)
        parser.add_argument("--text_encoder_dtype", type=str, default="bf16")
        parser.add_argument("--text_encoder_2_dtype", type=str, default="bf16")
        parser.add_argument("--text_encoder_3_dtype", type=str, default="bf16")
        parser.add_argument("--transformer_dtype", type=str, default="bf16")
        parser.add_argument("--vae_dtype", type=str, default="bf16")
        parser.add_argument("--layerwise_upcasting_modules", type=str, default=[], nargs="+", choices=["transformer"])
        parser.add_argument("--layerwise_upcasting_storage_dtype", type=str, default="float8_e4m3fn",
                            choices=["float8_e4m3fn", "float8_e5m2"])
        parser.add_argument("--layerwise_upcasting_skip_modules_pattern", type=str,
                            default=["patch_embed", "pos_embed", "x_embedder", "context_embedder", "^proj_in$",
                                     "^proj_out$", "norm"], nargs="+")
        parser.add_argument("--enable_slicing", action="store_true")
        parser.add_argument("--enable_tiling", action="store_true")
        
        parser.add_argument("--dino_feature_root", type=str, default=None)
        parser.add_argument("--vggt_feature_root", type=str, default=None)
        parser.add_argument("--flow_feature_root", type=str, default=None)
        parser.add_argument("--dino_in_channels", type=int, default=None)
        parser.add_argument("--dino_out_channels", type=int, default=None)
        parser.add_argument("--vggt_in_channels", type=int, default=None)
        parser.add_argument("--vggt_out_channels", type=int, default=None)
        parser.add_argument("--flow_in_channels", type=int, default=None)
        parser.add_argument("--flow_out_channels", type=int, default=None)

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        mapped_args.model_name = argparse_args.model_name
        mapped_args.pretrained_model_name_or_path = argparse_args.pretrained_model_name_or_path
        mapped_args.revision = argparse_args.revision
        mapped_args.variant = argparse_args.variant
        mapped_args.cache_dir = argparse_args.cache_dir
        mapped_args.tokenizer_id = argparse_args.tokenizer_id
        mapped_args.tokenizer_2_id = argparse_args.tokenizer_2_id
        mapped_args.tokenizer_3_id = argparse_args.tokenizer_3_id
        mapped_args.text_encoder_id = argparse_args.text_encoder_id
        mapped_args.text_encoder_2_id = argparse_args.text_encoder_2_id
        mapped_args.text_encoder_3_id = argparse_args.text_encoder_3_id
        mapped_args.transformer_id = argparse_args.transformer_id
        mapped_args.vae_id = argparse_args.vae_id
        mapped_args.text_encoder_dtype = _DTYPE_MAP[argparse_args.text_encoder_dtype]
        mapped_args.text_encoder_2_dtype = _DTYPE_MAP[argparse_args.text_encoder_2_dtype]
        mapped_args.text_encoder_3_dtype = _DTYPE_MAP[argparse_args.text_encoder_3_dtype]
        mapped_args.transformer_dtype = _DTYPE_MAP[argparse_args.transformer_dtype]
        mapped_args.vae_dtype = _DTYPE_MAP[argparse_args.vae_dtype]
        mapped_args.layerwise_upcasting_modules = argparse_args.layerwise_upcasting_modules
        mapped_args.layerwise_upcasting_storage_dtype = _DTYPE_MAP[argparse_args.layerwise_upcasting_storage_dtype]
        mapped_args.layerwise_upcasting_skip_modules_pattern = argparse_args.layerwise_upcasting_skip_modules_pattern
        mapped_args.enable_slicing = argparse_args.enable_slicing
        mapped_args.enable_tiling = argparse_args.enable_tiling
        
        mapped_args.dino_feature_root = getattr(argparse_args, "dino_feature_root", None)
        mapped_args.vggt_feature_root = getattr(argparse_args, "vggt_feature_root", None)
        mapped_args.flow_feature_root = getattr(argparse_args, "flow_feature_root", None)
        mapped_args.dino_in_channels = getattr(argparse_args, "dino_in_channels", None)
        mapped_args.dino_out_channels = getattr(argparse_args, "dino_out_channels", None)
        mapped_args.vggt_in_channels = getattr(argparse_args, "vggt_in_channels", None)
        mapped_args.vggt_out_channels = getattr(argparse_args, "vggt_out_channels", None)
        mapped_args.flow_in_channels = getattr(argparse_args, "flow_in_channels", None)
        mapped_args.flow_out_channels = getattr(argparse_args, "flow_out_channels", None)

    def validate_args(self, args: "BaseArgs"):
        pass


class InferenceArgs(ArgsConfigMixin):
    inference_type: InferenceType = InferenceType.TEXT_TO_VIDEO
    dataset_file: str = None

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--inference_type",
            type=str,
            default=InferenceType.TEXT_TO_VIDEO.value,
            choices=[x.value for x in InferenceType.__members__.values()],
        )
        parser.add_argument("--dataset_file", type=str, required=True)

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        mapped_args.inference_type = InferenceType(argparse_args.inference_type)
        mapped_args.dataset_file = argparse_args.dataset_file

    def validate_args(self, args: "BaseArgs"):
        pass


class AttentionProviderArgs(ArgsConfigMixin):
    attn_provider: AttentionProviderInference = "native"

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--attn_provider", type=str, default="native",
                            choices=["flash", "flash_varlen", "flex", "native", "_native_cudnn", "_native_efficient",
                                     "_native_flash", "_native_math", "sage", "sage_varlen",
                                     "_sage_qk_int8_pv_fp8_cuda", "_sage_qk_int8_pv_fp8_cuda_sm90",
                                     "_sage_qk_int8_pv_fp16_cuda", "_sage_qk_int8_pv_fp16_triton", "xformers"])

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        mapped_args.attn_provider = argparse_args.attn_provider

    def validate_args(self, args: "BaseArgs"):
        pass


class TorchConfigArgs(ArgsConfigMixin):
    compile_modules: List[str] = []
    compile_scopes: List[str] = None
    allow_tf32: bool = False
    float32_matmul_precision: str = "highest"

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--compile_modules", type=str, default=[], nargs="+")
        parser.add_argument("--compile_scopes", type=str, default=None, nargs="+")
        parser.add_argument("--allow_tf32", action="store_true")
        parser.add_argument("--float32_matmul_precision", type=str, default="highest",
                            choices=["highest", "high", "medium"])

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        compile_scopes = argparse_args.compile_scopes
        if len(argparse_args.compile_modules) > 0:
            if compile_scopes is None:
                compile_scopes = "regional"
            if isinstance(compile_scopes, list) and len(compile_scopes) == 1:
                compile_scopes = compile_scopes[0]
            if isinstance(compile_scopes, str):
                compile_scopes = [compile_scopes] * len(argparse_args.compile_modules)
        else:
            compile_scopes = []

        mapped_args.compile_modules = argparse_args.compile_modules
        mapped_args.compile_scopes = compile_scopes
        mapped_args.allow_tf32 = argparse_args.allow_tf32
        mapped_args.float32_matmul_precision = argparse_args.float32_matmul_precision

    def validate_args(self, args: "BaseArgs"):
        if len(args.compile_modules) > 0:
            assert len(args.compile_modules) == len(args.compile_scopes) and all(
                x in ["regional", "full"] for x in args.compile_scopes
            )


class MiscellaneousArgs(ArgsConfigMixin):
    seed: Optional[int] = None
    tracker_name: str = "finetune-inference"
    output_dir: str = None
    logging_dir: Optional[str] = "logs"
    init_timeout: int = 300
    nccl_timeout: int = 600
    report_to: str = "wandb"
    verbose: int = 1

    # [Modification] Register lora_path
    lora_path: Optional[str] = None

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--tracker_name", type=str, default="finetune")
        parser.add_argument("--output_dir", type=str, default="finetune-inference")
        parser.add_argument("--logging_dir", type=str, default="logs")
        parser.add_argument("--init_timeout", type=int, default=300)
        parser.add_argument("--nccl_timeout", type=int, default=600)
        parser.add_argument("--report_to", type=str, default="none", choices=["none", "wandb"])
        parser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2, 3])

        # [Modification] Add support for lora_path argument
        parser.add_argument("--lora_path", type=str, default=None, help="Path to custom weights or LoRA safetensors")

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        mapped_args.seed = argparse_args.seed
        mapped_args.tracker_name = argparse_args.tracker_name
        mapped_args.output_dir = argparse_args.output_dir
        mapped_args.logging_dir = argparse_args.logging_dir
        mapped_args.init_timeout = argparse_args.init_timeout
        mapped_args.nccl_timeout = argparse_args.nccl_timeout
        mapped_args.report_to = argparse_args.report_to
        mapped_args.verbose = argparse_args.verbose

        # [Modification] Map the provided lora_path
        mapped_args.lora_path = argparse_args.lora_path

    def validate_args(self, args: "BaseArgs"):
        pass


class BaseArgs:
    parallel_args = ParallelArgs()
    model_args = ModelArgs()
    inference_args = InferenceArgs()
    attention_provider_args = AttentionProviderArgs()
    torch_config_args = TorchConfigArgs()
    miscellaneous_args = MiscellaneousArgs()

    _registered_config_mixins: List[ArgsConfigMixin] = []
    _arg_group_map: Dict[str, ArgsConfigMixin] = {}

    def __init__(self):
        self._arg_group_map: Dict[str, ArgsConfigMixin] = {
            "parallel_args": self.parallel_args,
            "model_args": self.model_args,
            "inference_args": self.inference_args,
            "attention_provider_args": self.attention_provider_args,
            "torch_config_args": self.torch_config_args,
            "miscellaneous_args": self.miscellaneous_args,
        }
        for arg_config_mixin in self._arg_group_map.values():
            self.register_args(arg_config_mixin)

    def to_dict(self) -> Dict[str, Any]:
        arguments_to_dict = {}
        for config_mixin in self._registered_config_mixins:
            arguments_to_dict[config_mixin.__class__.__name__] = config_mixin.to_dict()
        return arguments_to_dict

    def register_args(self, config: ArgsConfigMixin) -> None:
        if not hasattr(self, "_extended_add_arguments"):
            self._extended_add_arguments = []
        self._extended_add_arguments.append((config.add_args, config.validate_args, config.map_args))
        self._registered_config_mixins.append(config)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        for extended_add_arg_fns in getattr(self, "_extended_add_arguments", []):
            add_fn, _, _ = extended_add_arg_fns
            add_fn(parser)

        args, remaining_args = parser.parse_known_args()
        logger.debug(f"Remaining unparsed arguments: {remaining_args}")

        for extended_add_arg_fns in getattr(self, "_extended_add_arguments", []):
            _, _, map_fn = extended_add_arg_fns
            map_fn(args, self)

        for extended_add_arg_fns in getattr(self, "_extended_add_arguments", []):
            _, validate_fn, _ = extended_add_arg_fns
            validate_fn(self)

        return self

    def __getattribute__(self, name: str):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            for arg_group in self._arg_group_map.values():
                if hasattr(arg_group, name):
                    return getattr(arg_group, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any):
        if name in self.__dict__:
            object.__setattr__(self, name, value)
            return
        for arg_group in self._arg_group_map.values():
            if hasattr(arg_group, name):
                setattr(arg_group, name, value)
                return
        object.__setattr__(self, name, value)


if __name__ == "__main__":
    main()