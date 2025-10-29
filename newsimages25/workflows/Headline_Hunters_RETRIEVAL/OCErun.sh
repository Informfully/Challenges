#!/bin/bash
set -e
CSV_PATH="../newsimages_25_v1.1/newsarticles.csv"
IMAGE_DIR="../newsimages_25_v1.1/newsimages"
OUT_DIR="./artifacts"

if [ ! -d "venv" ]; then
echo "[info] Creating virtual environment..."
python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip

echo "[info] Installing dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install open_clip_torch pillow tqdm pandas
pip install faiss-cpu #Use faiss-gpu if you have CUDA

echo "[info] Running OpenClipExhaustive.py..."
python3 OpenClipExhaustive.py --csv "$CSV_PATH" --image_dir "$IMAGE_DIR" --out_dir "$OUT_DIR"
echo "[done] Artifacts stored in $OUT_DIR"
