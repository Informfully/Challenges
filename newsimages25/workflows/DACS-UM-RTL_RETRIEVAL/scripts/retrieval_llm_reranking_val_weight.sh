WEIGHTS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

for weight in "${WEIGHTS[@]}"; do
python -m src.retrieval_pipeline_dataloader \
    --csv_path data/newsimages/newsimages_25_v1.0/newsarticles.csv \
    --image_path data/newsimages/newsimages_25_v1.0/newsimages \
    --model_family clip \
    --model_id openai/clip-vit-base-patch32 \
    --reranking_algorithm llm_rewriting \
    --num_candidates 10 \
    --llm_captions_path ./data/captions/llama38b_simple.json \
    --reranking_llm_weight $weight \
    --data_split val
done