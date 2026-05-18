#!/bin/bash
#SBATCH --job-name=zimage_caricature
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch-shared/%u/mediaeval/logs/zimage_%j.out
#SBATCH --error=/scratch-shared/%u/mediaeval/logs/zimage_%j.err

set -e
module load 2024 Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0
source /scratch-shared/$USER/mediaeval/env/caricature/bin/activate
export HF_HOME=/scratch-shared/$USER/mediaeval/models
cd /scratch-shared/$USER/mediaeval
python generate_zimage_caricature.py \
    --csv /scratch-shared/$USER/mediaeval/news_articles_test.csv
