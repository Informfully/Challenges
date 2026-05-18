# Caricature Baselines: Generating the Images

Scripts to generate the three caricature baselines (Klein, Qwen, Z-Image) for the MediaEval 2026 NewsImages baseline paper.

## Prerequisites

- Linux + NVIDIA GPU (A100 40GB or larger, or H100)
- Python 3.10+ with CUDA 12.4 / 12.6 runtime
- HuggingFace account with an access token
- Accept the license for `black-forest-labs/FLUX.2-klein-base-9B` in your browser (gated repo, must click "Agree" before downloading)

## Setup

```bash
# Create venv and install dependencies
python -m venv env
source env/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/huggingface/diffusers.git
pip install transformers accelerate sentencepiece protobuf huggingface_hub pillow safetensors peft

# Authenticate with HuggingFace
export HF_HOME=/scratch/$USER/mediaeval/models   # or wherever you want the cache
hf auth login

# Pre-download model weights (~75GB total)
hf download black-forest-labs/FLUX.2-klein-base-9B
hf download Qwen/Qwen-Image-Edit-2509
hf download lightx2v/Qwen-Image-Lightning \
    Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors
hf download Tongyi-MAI/Z-Image-Turbo
```

## Inputs

- `inputs/` — 50 baseline images (jpg or png) for Klein and Qwen
- `news_articles_test.csv` — test-set CSV with `article_id` and `article_title` columns, for Z-Image

## Generating images

### Klein and Qwen (image-to-image, 50 images each)

```bash
python generate_caricatures.py --model klein --input-dir inputs --output-dir outputs/KleinCaricature
python generate_caricatures.py --model qwen  --input-dir inputs --output-dir outputs/QwenCaricature
```

### Z-Image (text-to-image from headlines, ~800 images)

```bash
python generate_zimage_caricature.py --csv news_articles_test.csv --output-dir outputs/ZImageCaricature
```

Add `--limit N` to either script to do a quick test on the first N items before a full run.

## Running on a SLURM cluster

SLURM submission scripts are included for each pipeline. Edit the `module load` lines and `--partition` to match your cluster, then:

```bash
sbatch run_klein.sh
sbatch run_qwen.sh
sbatch run_zimage.sh
```

Expected wall-clock on A100 (40GB):

| Pipeline | Full run |
|---|---|
| Klein (50 images) | ~60–75 min |
| Qwen (50 images) | ~25–40 min |
| Z-Image (~800 images) | ~50–70 min |

## Outputs

PNG files at 460×260, named `<id>_Sotic_<approach>.png`, where `<approach>` is `KleinCaricature`, `QwenCaricature`, or `ZImageCaricature`.
