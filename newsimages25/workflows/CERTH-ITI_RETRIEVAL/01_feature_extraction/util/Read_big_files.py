from basic.bigfile import BigFile
from matplotlib import pyplot as plt
import numpy as np

def do_L2_norm(vec):
    L2_norm = np.linalg.norm(vec, 2)
    return 1.0 * np.array(vec) / L2_norm


visual_feat_path = '/home/dgalanop/Desktop/CERTH_VisualSearch_dualDense/TGIF_MSR_VTT_Activity_Vatex_2fps/FeatureData' \
                   '/resnetx101_imagenet13k,flatten0_output,os_resnet152_imagenet11k,flatten0_output,' \
                   'os_resnext101_32x16d_wsl,flatten0_output,os_CLIP_ViT_B_32_output,os'

visual_feats = BigFile(visual_feat_path)

vec = visual_feats.read_one(visual_feats.names[0])
print(visual_feats.names[0])

vec_res_101 = vec[:2048]
vec_res_152 = vec[2048:2048*2]
vec_res_wls = vec[2048*2:2048*3]
vec_res_clip = vec[2048*3:]
print()

fig, axs = plt.subplots(4)
fig.suptitle('Vertically stacked subplots')
axs[0].plot(vec_res_101)
axs[1].plot(vec_res_152)
axs[2].plot(vec_res_wls)
axs[3].plot(vec_res_clip)


fig, axs = plt.subplots(4)
fig.suptitle('Vertically stacked subplots')
axs[0].plot(do_L2_norm(vec_res_101))
axs[1].plot(do_L2_norm(vec_res_152))
axs[2].plot(do_L2_norm(vec_res_wls))
axs[3].plot(do_L2_norm(vec_res_clip))

