import numpy as np
import os
import torch
import tqdm

def main():
    year = 'mediaeval25' # 'mediaeval25' 'mediaeval25_imgCapti' 'mediaeval25_summ' 'mediaeval25_images'

    # Separate npy_files into tv22 and tv23
    tv22_folder = f'../data/{year}/'
    tv22_npy_files = [
        tv22_folder + f'{year}_beit3_base_patch16_384_coco_retrieval_similarities.npy',
        tv22_folder + f'{year}_beit3_base_patch16_384_flickr30k_similarities.npy',
        tv22_folder + f'{year}_beit3_large_patch16_384_coco_retrieval_similarities.npy',
        tv22_folder + f'{year}_beit3_large_patch16_384_flickr30k_similarities.npy',
    ]

    # Determine pre_trained_model
    pre_trained_model = 'beit3'

    # Process tv22 files
    errors_tv22 = []
    for npy_file in tv22_npy_files:
        error = np.load(npy_file)
        errors_tv22.append(error)
    errors_tv22 = np.stack(errors_tv22, axis=2)
    sum_array_tv22 = np.sum(errors_tv22, axis=2)
    norm_array_tv22 = sum_array_tv22 / np.linalg.norm(sum_array_tv22, axis=1, keepdims=True)

    # Save norm_array_tv22 to npy file
    savefile_tv22 = tv22_folder + year +'_' + pre_trained_model + '_L2_similarities.npy'
    np.save(savefile_tv22, norm_array_tv22)



if __name__ == "__main__":
    main()
