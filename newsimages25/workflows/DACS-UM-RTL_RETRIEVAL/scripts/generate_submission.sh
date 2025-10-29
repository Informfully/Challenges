for RERANKING in "identity" "llm_rewriting" "aesthetics" "vlm_judge"; do
  for MODEL in "clip-vit-base-patch32" "clip-vit-large-patch14" "siglip2-so400m-patch14-384"; do
    python src/create_submission_from_results.py \
      --results_dir ./results/inference/shards_250/$RERANKING/$MODEL/ \
      --submission_path ./submission/shards_250_resized/newsimages-um-rtl/
  done
done