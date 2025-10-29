python -m src.retrieval_pipeline_dataloader  \
    --csv_path data/newsimages/newsimages_25_v1.0/newsarticles.csv  \
    --image_path data/newsimages/newsimages_25_v1.0/newsimages     \
    --model_family clip     \
    --model_id openai/clip-vit-large-patch14 \
    --reranking_algorithm identity \
    --num_candidates 25 \
    --data_split val;

python -m src.retrieval_pipeline_dataloader  \
    --csv_path data/newsimages/newsimages_25_v1.0/newsarticles.csv  \
    --image_path data/newsimages/newsimages_25_v1.0/newsimages     \
    --model_family clip     \
    --model_id openai/clip-vit-large-patch14 \
    --reranking_algorithm aesthetics \
    --reranking_aesthetic_local_path aesthetic_checkpoints/sa_0_4_vit_l_14_linear.pth \
    --reranking_aesthetic_model_name ViT-L-14 \
    --num_candidates 25 \
    --data_split val \
    --reranking_aesthetic_weight 0.1;

python -m src.retrieval_pipeline_dataloader \
    --csv_path data/newsimages/newsimages_25_v1.0/newsarticles.csv \
    --image_path data/newsimages/newsimages_25_v1.0/newsimages \
    --model_family clip \
    --model_id openai/clip-vit-large-patch14 \
    --reranking_algorithm llm_rewriting \
    --num_candidates 25 \
    --llm_captions_path ./data/captions/llama38b_simple.json \
    --reranking_llm_weight 0.2 \
    --data_split val;

python -m src.retrieval_pipeline_dataloader  \
    --csv_path data/newsimages/newsimages_25_v1.0/newsarticles.csv  \
    --image_path data/newsimages/newsimages_25_v1.0/newsimages     \
    --model_family clip     \
    --model_id openai/clip-vit-large-patch14 \
    --reranking_algorithm vlm_judge \
    --reranking_config ./configs/vlm_reranking/qwen2_vl-7b.yaml \
    --num_candidates 25 \
    --reranking_vlm_prompt_path ./configs/vlm_reranking/prompt_score.txt \
    --data_split val \
    --reranking_vlm_weight 0.2;