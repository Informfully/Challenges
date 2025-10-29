python  BEIT3_img_feature_extraction__tqdm_Dataloader_CORRECT.py \
        --model beit3_large_patch16_384 \
        --input_size 384 \
        --task flickr30k \
        --batch_size 16 \
        --sentencepiece_model /home/aleventakis/PycharmProjects/LW_Network/BEIT/unilm-master/beit3/beit3.spm \
        --finetune /home/aleventakis/PycharmProjects/LW_Network/BEIT/unilm-master/beit3/checkpoints/beit3_large_patch16_384_f30k_retrieval.pth \
        --data_path /home/dgalanop/Desktop/Datasets/Vatex_FrameExtration_2fps/FrameExtration_2fps/Frames_OpenCV \
        --eval \
        --dist_eval

python  BEIT3_img_feature_extraction__tqdm_Dataloader_CORRECT.py \
        --model beit3_base_patch16_384 \
        --input_size 384 \
        --task coco_retrieval \
        --batch_size 16 \
        --sentencepiece_model /home/aleventakis/PycharmProjects/LW_Network/BEIT/unilm-master/beit3/beit3.spm \
        --finetune /home/aleventakis/PycharmProjects/LW_Network/BEIT/unilm-master/beit3/checkpoints/beit3_base_patch16_384_coco_retrieval.pth \
        --data_path /home/dgalanop/Desktop/Datasets/Vatex_FrameExtration_2fps/FrameExtration_2fps/Frames_OpenCV \
        --eval \
        --dist_eval
