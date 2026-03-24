# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import sys
from enum import Enum
from functools import partial
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils._pytree import tree_map

from dinov2.models import build_model_from_cfg
from dinov2.models.vision_transformer import DinoVisionTransformer
from dinov2.train.setup import setup
from dinov2.viz.attention import (
    call_m_softmax,
    get_attention_map_from_shape,
    get_image_from_url,
    get_vit_patch_size,
    make_attention_map,
    plot_attention_map,
    resize_attention_map,
)
from dinov2.viz.core import setup_and_build_model

logger = logging.getLogger("dinov2")


class ModelType(Enum):
    DINOV2 = "dinov2"
    DEIT = "deit"


def get_model_type(model: DinoVisionTransformer) -> ModelType:
    is_dinov2 = "layers" in dir(model.transformer.resblocks[0].attn)
    return ModelType.DINOV2 if is_dinov2 else ModelType.DEIT


def get_attention_map(
    model: DinoVisionTransformer,
    image: torch.Tensor,
    model_type: ModelType,
    apply_softmax: bool,
) -> torch.Tensor:
    attentions = model.get_last_selfattention(image)
    if apply_softmax:
        attentions = call_m_softmax(attentions)
    if model_type is ModelType.DINOV2:
        return attentions[:, :, 0, 1:]
    else:
        return attentions[:, :, 0, :]


def interpret_attention(
    model: DinoVisionTransformer,
    image: torch.Tensor,
    model_name: str,
    patch_size: int,
    device: str,
) -> torch.Tensor:
    model_type = get_model_type(model)
    logger.info(f"Model type: {model_type.value}")
    attention = get_attention_map(model, image, model_type, apply_softmax=True)

    if model_type is ModelType.DINOV2:
        image_size = tuple(image.shape[-2:])
        attention_map = get_attention_map_from_shape(
            attention,
            image_size,
            patch_size,
            "mean",
        )
    else:
        attention_map_shape = get_vit_patch_size(model_name)
        logger.info(f"Attention map shape: {attention_map_shape}")
        attention_map = make_attention_map(attention, attention_map_shape)

    attention = resize_attention_map(attention_map.unsqueeze(0), image.shape[-2:])
    return attention


def run(
    *,
    cfg,
    model: DinoVisionTransformer,
    model_name: str,
    patch_size: int,
    device: str,
    image: torch.Tensor,
    image_pil: Image,
    image_name: str,
    output_dir: str,
) -> None:
    logger.info("Interpret attention")

    attention = interpret_attention(model, image, model_name, patch_size, device)

    logger.info("Plot attention")

    plot_attention_map(
        image_pil,
        attention.squeeze(0),
        output_dir,
        image_name,
    )

    logger.info("Done")


def main(args: argparse.Namespace) -> None:
    logger.info("Starting")

    # Load image
    image_pil, image_tensor = get_image_from_url(args.image_path, args.image_size)

    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, patch_size, _ = setup_and_build_model(args)
    image_tensor = image_tensor.to(device)

    # Run
    run(
        cfg=args.opts,
        model=model,
        model_name=args.model_name,
        patch_size=patch_size,
        device=device,
        image=image_tensor,
        image_pil=image_pil,
        image_name=args.image_name,
        output_dir=args.output_dir,
    )


def get_args_parser(
    add_help: bool = True,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("DINOv2 attention visualization", add_help=add_help)
    parser.add_argument(
        "--config-file",
        default="/mnt/innovator/code/science/FineTrainer/data/dino/facebookresearch_dinov2_main/dinov2/configs/eval/vitg14_pretrain.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="vit_giant",
        help="Name of the model to load",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="https://dl.fbaipublicfiles.com/dinov2/images/shark.jpg",
        help="Path of the image to load",
    )
    parser.add_argument(
        "--image-name",
        type=str,
        default="attn",
        help="Name of the image to save",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs="+",
        default=[840, 1456],
        help="Size of the image to load",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Path of the output directory",
    )
    parser.add_argument(
        "--opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )

    return parser


def vit_attention(args: argparse.Namespace) -> None:
    main(args)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    vit_attention(args)