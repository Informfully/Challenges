from typing import Any, Dict, Optional

import torch
from transformers import (
    AlignModel,
    AlignProcessor,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    BlipModel,
    CLIPModel,
    Qwen2VLForConditionalGeneration,
    SiglipModel,
)

from src.models.align import AlignWrapper
from src.models.blip import BlipWrapper
from src.models.clip import CLIPWrapper
from src.models.jinaclip import JinaWrapper
from src.models.qwen2_vl import Qwen2VLWrapper
from src.models.siglip import SigLipWrapper
from src.models.siglip2 import SigLip2Wrapper
from src.models.smolvlm import SmolVLMWrapper

# model_id defines the default model_id that can be overwritten

CONFIGS = {
    "clip": {
        "model_id": "openai/clip-vit-base-patch32",
        "model_class": CLIPModel,
        "processor_class": AutoProcessor,
        "wrapper_class": CLIPWrapper,
    },
    "siglip": {
        "model_id": "google/siglip-base-patch16-256",
        "model_class": SiglipModel,
        "processor_class": AutoProcessor,
        "wrapper_class": SigLipWrapper,
    },
    "siglip_so": {
        "model_id": "google/siglip-so400m-patch14-384",
        "model_class": SiglipModel,
        "processor_class": AutoProcessor,
        "wrapper_class": SigLipWrapper,
    },
    "jinav2": {
        "model_id": "jinaai/jina-clip-v2",
        "model_class": AutoModel,
        "processor_class": AutoProcessor,
        "wrapper_class": JinaWrapper,
    },
    "siglip2": {
        "model_id": "google/siglip2-base-patch16-224",
        "model_class": SiglipModel,
        "processor_class": AutoProcessor,
        "wrapper_class": SigLip2Wrapper,
    },
    "siglip2_so": {
        "model_id": "google/siglip2-so400m-patch14-384",
        "model_class": SiglipModel,
        "processor_class": AutoProcessor,
        "wrapper_class": SigLip2Wrapper,
    },
    "blip": {
        "model_id": "Salesforce/blip-image-captioning-base",
        "model_class": BlipModel,
        "processor_class": AutoProcessor,
        "wrapper_class": BlipWrapper,
    },
    "align": {
        "model_id": "kakaobrain/align-base",
        "model_class": AlignModel,
        "processor_class": AlignProcessor,
        "wrapper_class": AlignWrapper,
    },
    "smolvlm": {
        "model_id": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "model_class": AutoModelForImageTextToText,
        "processor_class": AutoProcessor,
        "wrapper_class": SmolVLMWrapper,
    },
    "qwen2_vl": {
        "model_id": "Qwen/Qwen2-VL-7B-Instruct",
        "model_class": Qwen2VLForConditionalGeneration,
        "processor_class": AutoProcessor,
        "wrapper_class": Qwen2VLWrapper,
    },
}


def get_model_config(
        model_family: str,
        model_id: Optional[str] = None,
) -> Dict[str, Any]:
    config = CONFIGS.get(model_family, {})
    if not config:
        raise ValueError(f"Model config is not parsed. Plase use model_family from {list(CONFIGS.keys())}")
    if model_id is not None:
        config["model_id"] = model_id
    return config


def get_vlm_wrapper(model_family: str, model_id: str, device: str):
    """
    Initialize the VLM wrapper, model and processor.
    """
    model_config = get_model_config(model_family, model_id)
    model = model_config["model_class"].from_pretrained(
        model_config["model_id"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if model_family == "smolvlm" else "auto",
    )
    model = model.to(device)
    processor = model_config["processor_class"].from_pretrained(
        model_config["model_id"],
        trust_remote_code=True,
    )

    vlm_wrapper = model_config["wrapper_class"](model=model, processor=processor)
    return vlm_wrapper
