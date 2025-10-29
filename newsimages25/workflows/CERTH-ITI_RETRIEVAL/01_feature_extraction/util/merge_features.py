import os
import sys
import logging
from basic.constant import ROOT_PATH
from basic.generic_utils import Progbar
from basic.bigfile import BigFile
import numpy as np


def do_L2_norm(vec):
    L2_norm = np.linalg.norm(vec, 2)
    return 1.0 * np.array(vec) / L2_norm


def process(rootpath, collenction, features):

    feat_dir_1 = os.path.join(rootpath, collenction, 'FeatureData', features[0])
    featfile_1 = BigFile(feat_dir_1)
    shotNames_1 = featfile_1.names

    feat_dir_2 = os.path.join(rootpath, collenction, 'FeatureData', features[1])
    featfile_2 = BigFile(feat_dir_2)
    shotNames_2 = featfile_2.names

    print(">>> Process ")
    progbar = Progbar(len(shotNames_1))

    target_feat_dir = os.path.join(rootpath, collenction, 'FeatureData', '_'.join(features))
    if (not os.path.isdir(target_feat_dir)):
        os.mkdir(target_feat_dir)
    target_feat_file = os.path.join(target_feat_dir, 'id.feature.txt')

    with open(target_feat_file, 'w') as fw_feat:
        for name in shotNames_1:
            progbar.add(1)

            feat_1 = featfile_1.read_one(name)
            feat_2 = featfile_2.read_one(name)
            mergedlist = []
            mergedlist.extend(do_L2_norm(feat_1))
            mergedlist.extend(do_L2_norm(feat_2))
            # print len(feat_1)
            # print len(feat_2)
            fw_feat.write('%s %s\n' % (name, ' '.join(['%g' % x for x in mergedlist])))



# resnet152_imagenet11k,flatten0_output,os@resnext101_32x16d_wsl,flatten0_output,os@CLIP_ViT_B_32_output,os
def main(argv=None):
    rootpath = '/home/dgalanop/Desktop/CERTH_VisualSearch_dualDense/'
    collenction = 'MarineVideoKit'
    features = ['resnet152_imagenet11k,flatten0_output,os', 'resnext101_32x16d_wsl,flatten0_output,os', 'CLIP_ViT_B_32_output,os']
    return process(rootpath, collenction, features)


if __name__ == '__main__':
    sys.exit(main())
# python txt2bin.py 4096 /media/dgalanop/Toshiba_3TB/DCNN_Features/iacc.3/FeatureData/pyresnext-101_rbps13k,flatten0_output,os_pyresnet-152_imagenet11k,flatten0_output,os/id.feature.txt 0 /media/dgalanop/Toshiba_3TB/DCNN_Features/iacc.3/FeatureData/pyresnext-101_rbps13k,flatten0_output,os_pyresnet-152_imagenet11k,flatten0_output,os
