NUM_CANDIDATES=(25 50)

for num_candidates in "${NUM_CANDIDATES[@]}"; do
python -m src.retrieval_pipeline_dataloader \
    --csv_path data/newsimages/newsimages_25_v1.0/newsarticles.csv \
    --image_path data/newsimages/newsimages_25_v1.0/newsimages \
    --model_family clip \
    --model_id openai/clip-vit-base-patch32 \
    --reranking_algorithm llm_rewriting \
    --num_candidates $num_candidates \
    --llm_captions_path ./data/captions/llama38b_simple.json \
    --reranking_llm_weight 0.2 \
    --data_split val
done