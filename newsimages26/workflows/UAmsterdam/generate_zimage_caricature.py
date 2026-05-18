#!/usr/bin/env python
"""
Z-Image Turbo caricature baseline for MediaEval 2026 NewsImages.
Generates illustrations from article titles only (text-to-image).
Matches Lucien's Z-Image workflow specs; adds a caricature style suffix.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# --- Prompt template ---------------------------------------------------------

ZIMAGE_PROMPT_TEMPLATE = (
    "Editorial newspaper political cartoon illustration depicting: {title}. "
    "Hand-drawn ink linework, exaggerated facial features, bold black outlines, "
    "flat color blocks, simplified shapes, halftone shading, "
    "no photorealism, 2D illustration."
)

# --- Backbone config — matches Lucien's Z-Image workflow ---------------------

CONFIG = {
    "model_id": "Tongyi-MAI/Z-Image-Turbo",
    "num_inference_steps": 4,   # Lucien's KSampler steps
    "guidance_scale": 0.0,      # cfg=1.0 in ComfyUI == no guidance == 0.0 in diffusers
    "width": 1024,
    "height": 1024,
    "approach_name": "ZImageCaricature",
}

FINAL_W, FINAL_H = 460, 260
SEED = 42

# --- Pipeline ----------------------------------------------------------------

def load_pipeline():
    from diffusers import ZImagePipeline
    print("[zimage] loading Z-Image-Turbo (BF16)...", flush=True)
    pipe = ZImagePipeline.from_pretrained(
        CONFIG["model_id"],
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to("cuda")
    print("[zimage] loaded onto GPU", flush=True)
    return pipe


def generate(pipe, prompt):
    result = pipe(
        prompt=prompt,
        num_inference_steps=CONFIG["num_inference_steps"],
        guidance_scale=CONFIG["guidance_scale"],
        height=CONFIG["height"],
        width=CONFIG["width"],
        generator=torch.Generator("cuda").manual_seed(SEED),
    ).images[0]
    return result


def save_output(img, output_dir, article_id, approach):
    final = img.resize((FINAL_W, FINAL_H), Image.LANCZOS)
    out_path = output_dir / f"{article_id}_Sotic_{approach}.png"
    final.save(out_path, "PNG")
    return out_path


# --- Main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="Path to news_articles_test.csv")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N articles (for testing)")
    args = ap.parse_args()

    output_dir = (Path(args.output_dir) if args.output_dir
                  else Path(f"/scratch-shared/{os.environ['USER']}/mediaeval/outputs/"
                            f"{CONFIG['approach_name']}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"CSV: {args.csv}")
    print(f"Output dir: {output_dir}")

    # Load articles
    articles = []
    with open(args.csv, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = (row.get("article_title") or "").strip()
            article_id = (row.get("article_id") or "").strip()
            if not title or not article_id:
                continue
            articles.append({"article_id": article_id, "title": title})
            if args.limit and len(articles) >= args.limit:
                break

    print(f"Loaded {len(articles)} articles")
    if not articles:
        sys.exit("ERROR: no articles loaded from CSV")

    # Load pipeline
    pipe = load_pipeline()

    # Generate
    t0 = time.time()
    for i, art in enumerate(articles, 1):
        prompt = ZIMAGE_PROMPT_TEMPLATE.format(title=art["title"])
        title_preview = art["title"][:80].replace("\n", " ")
        print(f"[{i}/{len(articles)}] id={art['article_id']} | {title_preview}", flush=True)
        try:
            img = generate(pipe, prompt)
            out_path = save_output(img, output_dir, art["article_id"],
                                   CONFIG["approach_name"])
            print(f"    saved: {out_path.name}", flush=True)
        except Exception as e:
            print(f"    ERROR on {art['article_id']}: {e!r}", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone. {len(articles)} images in {elapsed:.0f}s "
          f"({elapsed/len(articles):.2f}s/img avg)")


if __name__ == "__main__":
    main()
