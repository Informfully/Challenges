#!/bin/bash
set -e
CSV_PATH="../newsimages_25_v1.1/newsarticles.csv"
IMAGE_DIR="../newsimages_25_v1.1/newsimages"
OUT_DIR="./artifacts"

if [ ! -f "$CSV_PATH" ]; then
  echo "[error] CSV file not found at $CSV_PATH"
  exit 1
fi

if [ ! -d "$IMAGE_DIR" ]; then
  echo "[error] Image directory not found at $IMAGE_DIR"
  exit 1
fi
if [ ! -d "venv" ]; then
  echo "[info] Creating virtual environment..."
  python3 -m venv venv
fi
source venv/bin/activate
echo "[info] Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install --force-reinstall --no-cache-dir open_clip_torch faiss-cpu pillow tqdm pandas
echo "[info] Running OpenClipSelective.py on CPU (safe mode)..."

export KMP_DUPLICATE_LIB_OK=TRUE
OMP_NUM_THREADS=1 python OpenClipSelective.py \
  --csv "$CSV_PATH" \
  --image_dir "$IMAGE_DIR" \
  --out_dir "$OUT_DIR" \
  --text_cols article_title article_tags \
  --image_id_col image_id \
  --id_col article_id \
  --model_name ViT-B-32 \
  --pretrained laion2b_s34b_b79k \
  --batch_size 4 \
  --device cpu \
  --num_workers 0
echo "[done] Artifacts stored in $OUT_DIR"
