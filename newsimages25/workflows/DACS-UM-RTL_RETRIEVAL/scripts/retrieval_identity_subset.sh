python -m src.retrieval_pipeline_dataloader \
    --csv_path data/newsimages_v1.1/newsarticles.csv \
    --image_path data/newsimages_v1.1/newsimages \
    --model_family clip \
    --model_id openai/clip-vit-large-patch14 \
    --reranking_algorithm identity \
    --debug \
    --subset_split data/newsimages_v1.1/subset.csv