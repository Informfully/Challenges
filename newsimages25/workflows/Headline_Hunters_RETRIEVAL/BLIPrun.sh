#!/usr/bin/env bash

set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 GROUP_NAME APPROACH_NAME"
  exit 1
fi

GROUP="$1"
APP="$2"
python3 BLIPrun.py \
--articles_csv ../newsimages_25_v1.1/newsarticles.csv \
--images_dir ../newsimages_25_v1.1/newsimages \
--weights weights.json \
--group_name "$GROUP" \
--approach_name "$APP" \
--subtask LARGE \
--out_dir ./ \
--device mps
echo "Run finished. Check .zip"
