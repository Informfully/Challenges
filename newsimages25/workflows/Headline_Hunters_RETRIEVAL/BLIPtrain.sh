#!/usr/bin/env bash
set -euo pipefail

python3 SubsetProcessing.py
python3 BLIPtrain.py \
--articles_csv ../newsimages_25_v1.1/newsarticles.csv \
--images_dir ../newsimages_25_v1.1/newsimages \
--subset_csv ./subset1.csv \
--out_weights weights.json \
--device mps \
--step 10
echo "Training finished. weights.json created."
