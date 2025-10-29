python -m src.write_faiss_index \
    --data data/yfcc100m/images/data/images/ \
    --output faiss/shards_250 \
    --model_family clip \
    --model_id openai/clip-vit-base-patch32 \
    --batch_size 32;

python -m src.write_faiss_index \
    --data data/yfcc100m/images/data/images/ \
    --output faiss/shards_250 \
    --model_family clip \
    --model_id openai/clip-vit-large-patch14 \
    --batch_size 32;

python -m src.write_faiss_index \
    --data data/yfcc100m/images/data/images/ \
    --output faiss/shards_250 \
    --model_family siglip2 \
    --model_id google/siglip2-so400m-patch14-384 \
    --batch_size 16