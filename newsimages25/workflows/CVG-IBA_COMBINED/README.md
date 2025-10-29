**Image Retrieval for Mediaeval Competition (NewsImages Task)**

1. Run faiss_index_pipeline.py in 'build' mode to build index.faiss and ids.npy using this command
   'python pipeline.py --mode build --csv Dataset/newsarticles.csv --images_dir Dataset/newsimages/ --index_path index.faiss --ids_path ids.npy'
2. Run finetuning_clip.py to finetune the clip model on newsiages provided by competition administrators.
3. Run retrieval_large.py to retrieve images for all 8500 articles listed in newsarticles.csv using the finetuned clip model's checkpoint.

4. Run retrieval_small.py to retrieve images for the subset of 30 articles listed in subset.csv.


