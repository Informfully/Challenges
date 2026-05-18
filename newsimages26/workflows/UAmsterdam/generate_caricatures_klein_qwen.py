#!/usr/bin/env python
"""Caricature baselines for MediaEval 2026 NewsImages."""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# --- Caricature prompts ------------------------------------------------------

KLEIN_PROMPT = (
    "transform this photograph into an editorial newspaper political cartoon, "
    "hand-drawn ink linework, exaggerated facial features, bold black outlines, "
    "flat color blocks, simplified shapes, halftone shading, "
    "no photorealism, 2D illustration"
)

QWEN_PROMPT = "turn this into a cartoon illustration"

# --- Backbone config matches Lucien's workflows ------------------------------

CONFIG = {
    "klein": {
        "model_id": "black-forest-labs/FLUX.2-klein-base-9B",
        "prompt": KLEIN_PROMPT,
        "guidance_scale": 5.0,
        "num_inference_steps": 20,
        "approach_name": "KleinCaricature",
    },
    "qwen": {
        "model_id": "Qwen/Qwen-Image-Edit-2509",
        "lora_repo": "lightx2v/Qwen-Image-Lightning",
        "lora_weight_name": "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
        "prompt": QWEN_PROMPT,
        "true_cfg_scale": 1.0,
        "num_inference_steps": 4,
        "negative_prompt": "",
        "approach_name": "QwenCaricature",
    },
}

TARGET_MEGAPIXELS = 1.0
DIM_MULTIPLE = 16
FINAL_W, FINAL_H = 460, 260
SEED = 42


def scale_to_megapixels(img, target_mp, multiple):
    w, h = img.size
    scale = math.sqrt((target_mp * 1_000_000) / (w * h))
    new_w = max(multiple, round(w * scale / multiple) * multiple)
    new_h = max(multiple, round(h * scale / multiple) * multiple)
    return img.resize((new_w, new_h), Image.LANCZOS)


def load_image_safe(path):
    img = Image.open(path).convert("RGB")
    return scale_to_megapixels(img, TARGET_MEGAPIXELS, DIM_MULTIPLE)


def save_output(img, output_dir, stem, approach):
    final = img.resize((FINAL_W, FINAL_H), Image.LANCZOS)
    out_path = output_dir / f"{stem}_Sotic_{approach}.png"
    final.save(out_path, "PNG")
    return out_path


def load_klein():
    from diffusers import Flux2KleinPipeline
    print("[klein] loading FLUX.2-klein-base-9B (BF16, ~40GB w/ text encoder)...", flush=True)
    pipe = Flux2KleinPipeline.from_pretrained(
        CONFIG["klein"]["model_id"],
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_sequential_cpu_offload()
    return pipe


def load_qwen():
    from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler

    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    print("[qwen] loading Qwen-Image-Edit-2509...", flush=True)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        CONFIG["qwen"]["model_id"],
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
    )
    print("[qwen] loading Lightning 4-step LoRA...", flush=True)
    pipe.load_lora_weights(
        CONFIG["qwen"]["lora_repo"],
        weight_name=CONFIG["qwen"]["lora_weight_name"],
    )
    pipe.enable_sequential_cpu_offload()
    return pipe


def generate_klein(pipe, input_image):
    cfg = CONFIG["klein"]
    w, h = input_image.size
    result = pipe(
        prompt=cfg["prompt"],
        image=[input_image],
        height=h,
        width=w,
        guidance_scale=cfg["guidance_scale"],
        num_inference_steps=cfg["num_inference_steps"],
        generator=torch.Generator(device="cuda").manual_seed(SEED),
    ).images[0]
    return result


def generate_qwen(pipe, input_image):
    cfg = CONFIG["qwen"]
    with torch.inference_mode():
        result = pipe(
            image=[input_image],
            prompt=cfg["prompt"],
            generator=torch.manual_seed(SEED),
            true_cfg_scale=cfg["true_cfg_scale"],
            negative_prompt=cfg["negative_prompt"],
            num_inference_steps=cfg["num_inference_steps"],
            num_images_per_prompt=1,
        ).images[0]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["klein", "qwen"], required=True)
    ap.add_argument("--input-dir",
                    default=f"/scratch-shared/{os.environ['USER']}/mediaeval/inputs")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = CONFIG[args.model]
    input_dir = Path(args.input_dir)
    output_dir = (Path(args.output_dir) if args.output_dir
                  else Path(f"/scratch-shared/{os.environ['USER']}/mediaeval/outputs/"
                            f"{cfg['approach_name']}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model}")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Prompt: {cfg['prompt']!r}")

    if args.model == "klein":
        pipe = load_klein()
        gen_fn = generate_klein
    else:
        pipe = load_qwen()
        gen_fn = generate_qwen

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in exts])
    print(f"Found {len(image_paths)} input images")
    if not image_paths:
        sys.exit("ERROR: no input images found")

    t0 = time.time()
    for i, p in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] {p.name}", flush=True)
        try:
            img_in = load_image_safe(p)
            img_out = gen_fn(pipe, img_in)
            out_path = save_output(img_out, output_dir, p.stem, cfg["approach_name"])
            print(f"    saved: {out_path.name}", flush=True)
        except Exception as e:
            print(f"    ERROR on {p.name}: {e!r}", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone. {len(image_paths)} images in {elapsed:.0f}s "
          f"({elapsed/len(image_paths):.1f}s/img avg)")


if __name__ == "__main__":
    main()
