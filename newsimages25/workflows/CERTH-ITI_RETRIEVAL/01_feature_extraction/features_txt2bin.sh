## CLIP
dim=512
featurefile=/path/to/features/dataset_CLIP_RN101_feas.txt
resultdir=/path/to/features/dataset/FeatureData/CLIP_RN101/
python simpleknn/txt2bin.py $dim $featurefile 0 $resultdir --overwrite 1
python util/get_frameInfo.py --collection YFCC100M --feature CLIP_RN101 --rootpath /m2/YFCC100M_Features/

dim=1024
featurefile=/path/to/features/dataset_CLIP_RN50_feas.txt
resultdir=/path/to/features/dataset/FeatureData/CLIP_RN50/
python simpleknn/txt2bin.py $dim $featurefile 0 $resultdir --overwrite 1
python util/get_frameInfo.py --collection YFCC100M --feature CLIP_RN50 --rootpath /m2/YFCC100M_Features/

dim=768
featurefile=/path/to/features/dataset_CLIP_RN50x16_feas.txt
resultdir=/path/to/features/dataset/FeatureData/CLIP_RN50x16/
python simpleknn/txt2bin.py $dim $featurefile 0 $resultdir --overwrite 1
python util/get_frameInfo.py --collection YFCC100M --feature CLIP_RN50x16 --rootpath /m2/YFCC100M_Features/

dim=640
featurefile=/path/to/features/dataset_CLIP_RN50x4_feas.txt
resultdir=/path/to/features/dataset/FeatureData/CLIP_RN50x4/
python simpleknn/txt2bin.py $dim $featurefile 0 $resultdir --overwrite 1
python util/get_frameInfo.py --collection YFCC100M --feature CLIP_RN50x4 --rootpath /m2/YFCC100M_Features/

dim=1024
featurefile=/path/to/features/dataset_CLIP_RN50x64_feas.txt
resultdir=/path/to/features/dataset/FeatureData/CLIP_RN50x64/
python simpleknn/txt2bin.py $dim $featurefile 0 $resultdir --overwrite 1
python util/get_frameInfo.py --collection YFCC100M --feature CLIP_RN50x64 --rootpath /m2/YFCC100M_Features/

dim=512
featurefile=/path/to/features/dataset_CLIP_ViT_B_16_feas.txt
resultdir=/path/to/features/dataset/FeatureData/CLIP_ViT_B_16/
python simpleknn/txt2bin.py $dim $featurefile 0 $resultdir --overwrite 1
python util/get_frameInfo.py --collection YFCC100M --feature CLIP_ViT_B_16 --rootpath /m2/YFCC100M_Features/

dim=512
featurefile=/path/to/features/dataset_CLIP_ViT_B_32_feas.txt
resultdir=/path/to/features/dataset/FeatureData/CLIP_ViT_B_32/
python simpleknn/txt2bin.py $dim $featurefile 0 $resultdir --overwrite 1
python util/get_frameInfo.py --collection YFCC100M --feature CLIP_ViT_B_32 --rootpath /m2/YFCC100M_Features/

dim=768
featurefile=/path/to/features/dataset_CLIP_ViT_L_14_feas.txt
resultdir=/path/to/features/dataset/FeatureData/CLIP_ViT_L_14/
python simpleknn/txt2bin.py $dim $featurefile 0 $resultdir --overwrite 1
python util/get_frameInfo.py --collection YFCC100M --feature CLIP_ViT_L_14 --rootpath /m2/YFCC100M_Features/



## SLIP
dim=512
featurefile=/path/to/features/dataset_Slip_base_feas.txt
resultdir=/path/to/features/dataset/FeatureData/SLIP_base/
python simpleknn/txt2bin.py $dim $featurefile 0 $resultdir --overwrite 1
python util/get_frameInfo.py --collection YFCC100M --feature SLIP_base --rootpath /m2/YFCC100M_Features/

dim=512
featurefile=/path/to/features/dataset_Slip_base_CC12M_feas.txt
resultdir=/path/to/features/dataset/FeatureData/SLIP_base_CC12M/
python simpleknn/txt2bin.py $dim $featurefile 0 $resultdir --overwrite 1
python util/get_frameInfo.py --collection YFCC100M --feature SLIP_base_CC12M --rootpath /m2/YFCC100M_Features/

dim=512
featurefile=/path/to/features/dataset_Slip_base_CC3M_feas.txt
resultdir=/path/to/features/dataset/FeatureData/SLIP_base_CC3M/
python simpleknn/txt2bin.py $dim $featurefile 0 $resultdir --overwrite 1
python util/get_frameInfo.py --collection YFCC100M --feature SLIP_base_CC3M --rootpath /m2/YFCC100M_Features/

dim=512
featurefile=/path/to/features/dataset_Slip_large_feas.txt
resultdir=/path/to/features/dataset/FeatureData/SLIP_large/
python simpleknn/txt2bin.py $dim $featurefile 0 $resultdir --overwrite 1
python util/get_frameInfo.py --collection YFCC100M --feature SLIP_large --rootpath /m2/YFCC100M_Features/

dim=512
featurefile=/path/to/features/dataset_Slip_small_feas.txt
resultdir=/path/to/features/dataset/FeatureData/SLIP_small/
python simpleknn/txt2bin.py $dim $featurefile 0 $resultdir --overwrite 1
python util/get_frameInfo.py --collection YFCC100M --feature SLIP_small --rootpath /m2/YFCC100M_Features/


## BEIT3
dim=768
featurefile=/path/to/features/dataset_beit3_base_patch16_384_coco_retrieval_features_mediaeval25.txt
resultdir=/path/to/features/dataset/FeatureData/beit3_base_patch16_384_coco_retrievalv/
python simpleknn/txt2bin.py $dim $featurefile 0 $resultdir
python util/get_frameInfo.py --collection YFCC100M --feature beit3_base_patch16_384_coco_retrievalv --rootpath /m2/YFCC100M_Features/

dim=768
featurefile=/path/to/features/dataset_beit3_base_patch16_384_flickr30k_features_mediaeval25.txt
resultdir=/path/to/features/dataset/FeatureData/beit3_base_patch16_384_f30k_retrieval/
python simpleknn/txt2bin.py $dim $featurefile 0 $resultdir
python util/get_frameInfo.py --collection YFCC100M --feature beit3_base_patch16_384_f30k_retrieval --rootpath /m2/YFCC100M_Features/

dim=1024
featurefile=/path/to/features/dataset_beit3_large_patch16_384_coco_retrieval_features_mediaeval25.txt
resultdir=/path/to/features/dataset/FeatureData/beit3_large_patch16_384_coco_retrieval/
python simpleknn/txt2bin.py $dim $featurefile 0 $resultdir
python util/get_frameInfo.py --collection YFCC100M --feature beit3_large_patch16_384_coco_retrieval --rootpath /m2/YFCC100M_Features/

dim=1024
featurefile=/path/to/features/dataset_beit3_large_patch16_384_flickr30k_features_mediaeval25.txt
resultdir=/path/to/features/dataset/FeatureData/beit3_large_patch16_384_f30k_retrieval/
python simpleknn/txt2bin.py $dim $featurefile 0 $resultdir
python util/get_frameInfo.py --collection YFCC100M --feature beit3_large_patch16_384_f30k_retrieval --rootpath /m2/YFCC100M_Features/


#BLIP
dim=256
featurefile=/path/to/features/dataset_BLIP_base_coco_feas.txt
resultdir=/path/to/features/dataset/FeatureData/BLIP_itc_model_base_retrieval_coco/
python simpleknn/txt2bin.py $dim $featurefile 0 $resultdir
python util/get_frameInfo.py --collection YFCC100M --feature BLIP_itc_model_base_retrieval_coco --rootpath /m2/YFCC100M_Features/

dim=256
featurefile=/path/to/features/dataset_BLIP_base_flickr_feas.txt
resultdir=/path/to/features/dataset/FeatureData/BLIP_itc_model_base_retrieval_flickr/
python simpleknn/txt2bin.py $dim $featurefile 0 $resultdir
python util/get_frameInfo.py --collection YFCC100M --feature BLIP_itc_model_base_retrieval_flickr --rootpath /m2/YFCC100M_Features/

dim=256
featurefile=/path/to/features/dataset_BLIP_large_coco_feas.txt
resultdir=/path/to/features/dataset/FeatureData/BLIP_itc_model_large_retrieval_coco/
python simpleknn/txt2bin.py $dim $featurefile 0 $resultdir
python util/get_frameInfo.py --collection YFCC100M --feature BLIP_itc_model_large_retrieval_coco --rootpath /m2/YFCC100M_Features/

dim=256
featurefile=/path/to/features/dataset_BLIP_large_flickr_feas.txt
resultdir=/path/to/features/dataset/FeatureData/BLIP_itc_model_large_retrieval_flickr/
python simpleknn/txt2bin.py $dim $featurefile 0 $resultdir
python util/get_frameInfo.py --collection YFCC100M --feature BLIP_itc_model_large_retrieval_flickr --rootpath /m2/YFCC100M_Features/

 

#BLIP2
dim=256
featurefile=/path/to/features/dataset_blip2_pretrain_feas.txt
resultdir=/path/to/features/dataset/FeatureData/BLIP2_itm_pretrain/
python simpleknn/txt2bin.py $dim $featurefile 0 $resultdir
python util/get_frameInfo.py --collection YFCC100M --feature BLIP2_itm_pretrain --rootpath /m2/YFCC100M_Features/

dim=256
featurefile=/path/to/features/dataset_blip2_pretrain_vitL.txt
resultdir=/path/to/features/dataset/FeatureData/BLIP2_itm_pretrain_vitL/
python simpleknn/txt2bin.py $dim $featurefile 0 $resultdir
python util/get_frameInfo.py --collection YFCC100M --feature BLIP2_itm_pretrain_vitL --rootpath /m2/YFCC100M_Features/

dim=256
featurefile=/path/to/features/dataset_blip2_coco_feas.txt
resultdir=/path/to/features/dataset/FeatureData/BLIP2_itm_coco/
python simpleknn/txt2bin.py $dim $featurefile 0 $resultdir
python util/get_frameInfo.py --collection YFCC100M --feature BLIP2_itm_coco --rootpath /m2/YFCC100M_Features/
