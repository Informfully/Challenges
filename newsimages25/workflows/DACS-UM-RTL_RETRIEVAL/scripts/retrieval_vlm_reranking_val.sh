WEIGHTS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

for weight in ${WEIGHTS[@]}; do
    python -m src.retrieval_pipeline_dataloader  \
    --csv_path data/newsimages/newsimages_25_v1.0/newsarticles.csv  \
    --image_path data/newsimages/newsimages_25_v1.0/newsimages     \
    --model_family clip     \
    --model_id openai/clip-vit-base-patch32 \
    --reranking_algorithm vlm_judge \
    --reranking_config ./configs/vlm_reranking/qwen2_vl-7b.yaml \
    --num_candidates 10 \
    --reranking_vlm_prompt_path ./configs/vlm_reranking/prompt_score.txt \
    --data_split val \
    --reranking_vlm_weight $weight
done