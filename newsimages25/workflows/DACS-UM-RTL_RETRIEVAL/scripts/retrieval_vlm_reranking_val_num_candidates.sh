NUM_CANDIDATES=(25 50)

for num_candidates in ${NUM_CANDIDATES[@]}; do
    python -m src.retrieval_pipeline_dataloader  \
    --csv_path data/newsimages/newsimages_25_v1.0/newsarticles.csv  \
    --image_path data/newsimages/newsimages_25_v1.0/newsimages     \
    --model_family clip     \
    --model_id openai/clip-vit-base-patch32 \
    --reranking_algorithm vlm_judge \
    --reranking_config ./configs/vlm_reranking/qwen2_vl-7b.yaml \
    --num_candidates $num_candidates \
    --reranking_vlm_prompt_path ./configs/vlm_reranking/prompt_score.txt \
    --data_split val \
    --reranking_vlm_weight 0.2
done