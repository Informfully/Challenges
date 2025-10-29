python BLIP2_MediaEval2025.py --year mediaeval25
python BLIP_itc_MediaEval2025.py --year mediaeval25
python SLIP_mediaEval2025.py --year mediaeval25
python CLIP_mediaEval2025.py --year mediaeval25


python BEIT3_MediaEval2025.py --model beit3_large_patch16_384 --input_size 384 --task coco_retrieval --batch_size 16 --sentencepiece_model beit3/beit3.spm --finetune beit3/checkpoints/beit3_base_patch16_384_f30k_retrieval.pth --eval --dist_eval --year mediaeval25
python BEIT3_MediaEval2025.py --model beit3_large_patch16_384 --input_size 384 --task flickr30k --batch_size 16 --sentencepiece_model beit3/beit3.spm --finetune beit3/checkpoints/beit3_base_patch16_384_f30k_retrieval.pth --eval --dist_eval --year mediaeval25
python BEIT3_MediaEval2025.py --model beit3_base_patch16_384 --input_size 384 --task coco_retrieval --batch_size 16 --sentencepiece_model beit3/beit3.spm --finetune beit3/checkpoints/beit3_large_patch16_384_f30k_retrieval.pth --eval --dist_eval --year mediaeval25
python BEIT3_MediaEval2025.py --model beit3_base_patch16_384 --input_size 384 --task flickr30k --batch_size 16 --sentencepiece_model beit3/beit3.spm --finetune beit3/checkpoints/beit3_large_patch16_384_f30k_retrieval.pth --eval --dist_eval --year mediaeval25
python BEIT3_MediaEval2025_combine.py

python MediaEval2025_combine_similarities.py