python -m src.retrieval_pipeline_inference \
   --faiss_index_path ./faiss/shards_250/google/siglip2-so400m-patch14-384/image_index.faiss \
   --image_paths_file ./faiss/shards_250/google/siglip2-so400m-patch14-384/image_paths.txt \
   --csv_path ./data/newsimages_v1.1/newsarticles.csv \
   --subset_split ./data/newsimages_v1.1/subset.csv \
   --model_family siglip2 \
   --model_id google/siglip2-so400m-patch14-384 \
   --output_path ./results/inference/shards_250/identity/siglip2-so400m-patch14-384 \
   --num_candidates 25;

python -m src.retrieval_pipeline_inference \
   --faiss_index_path ./faiss/shards_250/google/siglip2-so400m-patch14-384/image_index.faiss \
   --image_paths_file ./faiss/shards_250/google/siglip2-so400m-patch14-384/image_paths.txt \
   --csv_path ./data/newsimages_v1.1/newsarticles.csv \
   --subset_split ./data/newsimages_v1.1/subset.csv \
   --model_family siglip2 \
   --model_id google/siglip2-so400m-patch14-384 \
   --output_path ./results/inference/shards_250/vlm_judge/siglip2-so400m-patch14-384 \
   --num_candidates 25 \
   --reranking_algorithm vlm_judge \
   --reranking_config ./configs/vlm_reranking/qwen2_vl-7b.yaml \
   --reranking_vlm_prompt_path ./configs/vlm_reranking/prompt_score.txt \
   --reranking_vlm_weight 0.2;

python -m src.retrieval_pipeline_inference \
   --faiss_index_path ./faiss/shards_250/google/siglip2-so400m-patch14-384/image_index.faiss \
   --image_paths_file ./faiss/shards_250/google/siglip2-so400m-patch14-384/image_paths.txt \
   --csv_path ./data/newsimages_v1.1/newsarticles.csv \
   --subset_split ./data/newsimages_v1.1/subset.csv \
   --model_family siglip2 \
   --model_id google/siglip2-so400m-patch14-384 \
   --output_path ./results/inference/shards_250/llm_rewriting/siglip2-so400m-patch14-384 \
   --num_candidates 25 \
   --reranking_algorithm llm_rewriting \
   --llm_captions_path ./data/captions/llama38b_simple.json \
   --reranking_llm_weight 0.2;

python -m src.retrieval_pipeline_inference \
   --faiss_index_path ./faiss/shards_250/google/siglip2-so400m-patch14-384/image_index.faiss \
   --image_paths_file ./faiss/shards_250/google/siglip2-so400m-patch14-384/image_paths.txt \
   --csv_path ./data/newsimages_v1.1/newsarticles.csv \
   --subset_split ./data/newsimages_v1.1/subset.csv \
   --model_family siglip2 \
   --model_id google/siglip2-so400m-patch14-384 \
   --output_path ./results/inference/shards_250/aesthetics/siglip2-so400m-patch14-384 \
   --num_candidates 25 \
   --reranking_algorithm aesthetics \
   --reranking_aesthetic_local_path aesthetic_checkpoints/sa_0_4_vit_l_14_linear.pth \
   --reranking_aesthetic_model_name ViT-L-14 \
   --reranking_aesthetic_weight 0.2;