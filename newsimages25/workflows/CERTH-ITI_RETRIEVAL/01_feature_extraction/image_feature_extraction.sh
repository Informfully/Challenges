python BLIP2_feature_extraction_mediaEval.py dataset_paths.txt dataset_blip2_pretrain_vitL.txt --model 'pretrain_vitL'
python BLIP2_feature_extraction_mediaEval.py dataset_paths.txt dataset_blip2_pretrain_feas.txt --model 'pretrain'
python BLIP2_feature_extraction_mediaEval.py dataset_paths.txt dataset_blip2_coco_feas.txt --model 'coco'

python BLIP_feature_extraction_itc_mediaEval.py dataset_paths.txt dataset_BLIP_base_coco_feas.txt --model 'model_base_retrieval_coco'
python BLIP_feature_extraction_itc_mediaEval.py dataset_paths.txt dataset_BLIP_base_flickr_feas.txt --model 'model_base_retrieval_flickr'
python BLIP_feature_extraction_itc_mediaEval.py dataset_paths.txt dataset_BLIP_large_coco_feas.txt --model 'model_large_retrieval_coco'
python BLIP_feature_extraction_itc_mediaEval.py dataset_paths.txt dataset_BLIP_large_flickr_feas.txt --model 'model_large_retrieval_flickr'

python SLIP_feature_extraction_mediaEval.py dataset_paths.txt dataset_Slip_small_feas.txt --model 'slip_small'
python SLIP_feature_extraction_mediaEval.py dataset_paths.txt dataset_Slip_base_feas.txt --model 'slip_base'
python SLIP_feature_extraction_mediaEval.py dataset_paths.txt dataset_Slip_large_feas.txt --model 'slip_large'
python SLIP_feature_extraction_mediaEval.py dataset_paths.txt dataset_Slip_base_CC3M_feas.txt --model 'slip_base_CC3M'
python SLIP_feature_extraction_mediaEval.py dataset_paths.txt dataset_Slip_base_CC12M_feas.txt --model 'slip_base_CC12M'

python CLIP_feature_extraction_mediaEval.py dataset_paths.txt dataset_CLIP_RN50x4_feas.txt --model RN50x4
python CLIP_feature_extraction_mediaEval.py dataset_paths.txt dataset_CLIP_RN50x16_feas.txt --model RN50x16
python CLIP_feature_extraction_mediaEval.py dataset_paths.txt dataset_CLIP_RN50x64_feas.txt --model RN50x64
python CLIP_feature_extraction_mediaEval.py dataset_paths.txt dataset_CLIP_RN50_feas.txt --model RN50
python CLIP_feature_extraction_mediaEval.py dataset_paths.txt dataset_CLIP_RN101_feas.txt --model RN101
python CLIP_feature_extraction_mediaEval.py dataset_paths.txt dataset_CLIP_ViT_B_16_feas.txt --model 'ViT-B/16'
python CLIP_feature_extraction_mediaEval.py dataset_paths.txt dataset_CLIP_ViT_B_32_feas.txt --model 'ViT-B/32'
python CLIP_feature_extraction_mediaEval.py dataset_paths.txt dataset_CLIP_ViT_L_14_feas.txt --model 'ViT-L/14'


python BEIT3_feature_extraction_mediaEval.py --model beit3_base_patch16_384 --input_size 384 --task coco_retrieval --batch_size 128 --sentencepiece_model beit3/beit3.spm --finetune ../models/beit3/checkpoints/beit3_base_patch16_384_f30k_retrieval.pth --eval --dist_eval --output_dir ./ --keyframe_path_file dataset_paths.txt
python BEIT3_feature_extraction_mediaEval.py --model beit3_base_patch16_384 --input_size 384 --task flickr30k --batch_size 128 --sentencepiece_model beit3/beit3.spm --finetune ../models/beit3/checkpoints/beit3_base_patch16_384_f30k_retrieval.pth --eval --dist_eval --output_dir ./ --keyframe_path_file dataset_paths.txt
python BEIT3_feature_extraction_mediaEval.py --model beit3_large_patch16_384 --input_size 384 --task coco_retrieval --batch_size 128 --sentencepiece_model beit3/beit3.spm --finetune ../models/beit3/checkpoints/beit3_large_patch16_384_f30k_retrieval.pth --eval --dist_eval --output_dir ./ --keyframe_path_file dataset_paths.txt
python BEIT3_feature_extraction_mediaEval.py --model beit3_large_patch16_384 --input_size 384 --task flickr30k --batch_size 16 --sentencepiece_model beit3/beit3.spm --finetune ../models/beit3/checkpoints/beit3_large_patch16_384_f30k_retrieval.pth --eval --dist_eval --output_dir ./ --keyframe_path_file dataset_paths.txt
