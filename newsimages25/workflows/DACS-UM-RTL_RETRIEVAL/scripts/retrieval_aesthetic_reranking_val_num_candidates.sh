NUM_CANDIDATES=(25 50)

for num_candidates in ${NUM_CANDIDATES[@]}; do
    python -m src.retrieval_pipeline_dataloader  \
    --csv_path data/newsimages/newsimages_25_v1.0/newsarticles.csv  \
    --image_path data/newsimages/newsimages_25_v1.0/newsimages     \
    --model_family clip     \
    --model_id openai/clip-vit-base-patch32 \
    --reranking_algorithm aesthetics \
    --reranking_aesthetic_local_path aesthetic_checkpoints/sa_0_4_vit_l_14_linear.pth \
    --reranking_aesthetic_model_name ViT-L-14 \
    --num_candidates $num_candidates \
    --data_split val \
    --reranking_aesthetic_weight 0.1
done