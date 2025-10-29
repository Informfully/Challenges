python -m src.retrieval_pipeline_dataloader \
    --csv_path data/newsimages/newsimages_25_v1.0/newsarticles.csv \
    --image_path data/newsimages/newsimages_25_v1.0/newsimages \
    --model_family clip \
    --model_id openai/clip-vit-base-patch32 \
    --reranking_algorithm identity \
    --log_retrieved_images ./results/development/identity/clip-vit-base-patch32 \
    --num_candidates 100 \
    --batch_size 16;

python -m src.retrieval_pipeline_dataloader \
    --csv_path data/newsimages/newsimages_25_v1.0/newsarticles.csv \
    --image_path data/newsimages/newsimages_25_v1.0/newsimages \
    --model_family clip \
    --model_id openai/clip-vit-large-patch14 \
    --reranking_algorithm identity \
    --log_retrieved_images ./results/development/identity/clip-vit-large-patch14 \
    --num_candidates 100 \
    --batch_size 16;

python -m src.retrieval_pipeline_dataloader \
    --csv_path data/newsimages/newsimages_25_v1.0/newsarticles.csv \
    --image_path data/newsimages/newsimages_25_v1.0/newsimages \
    --model_family siglip2 \
    --model_id google/siglip2-base-patch16-224 \
    --reranking_algorithm identity \
    --log_retrieved_images ./results/development/identity/siglip2-base-patch16-224 \
    --num_candidates 100 \
    --batch_size 16;