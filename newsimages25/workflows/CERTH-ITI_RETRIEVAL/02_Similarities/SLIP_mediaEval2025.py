
import torch
from PIL import Image
import os
import sys
import numpy as np
# from scipy.spatial import distance
from torchvision import transforms

import tqdm
import time
# from scipy.special import softmax

from basic.bigfile import BigFile
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import torch.nn as nn
import sys
import time
import os
import pandas as pd

from PIL import Image
import torch
from torchvision import transforms
import os.path

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image

import json
import argparse
from collections import OrderedDict
from SLIP.SLIP_main.tokenizer import SimpleTokenizer

import SLIP.SLIP_main.models as SLIP_models
import SLIP.SLIP_main.utils as utils


def do_L2_norm(vec):
    L2_norm = np.linalg.norm(vec, 2)
    return 1.0 * np.array(vec) / L2_norm


class Dataset4DualEncoding(torch.utils.data.Dataset):
    """
       Load captions and video frame features by pre-trained CNN model.
       """
    def __init__(self,  visual_feat, do_visual_feas_norm, video2frames=None):

        # self.video_ids = set()
        self.video2frames = video2frames
        self.do_visual_feas_norm = do_visual_feas_norm

        self.video_ids = [key for key in self.video2frames.keys()]
        self.visual_feat = visual_feat

        self.length = len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]

        # video
        frame_list = self.video2frames[video_id]

        frame_vecs = []
        for frame_id in frame_list:
            # l_2 normalize
            if (self.do_visual_feas_norm):
                frame_vecs.append(do_L2_norm(self.visual_feat.read_one(frame_id)))
            else:
                frame_vecs.append(self.visual_feat.read_one(frame_id))
        frames_tensor = torch.Tensor(np.array(frame_vecs))

        return frames_tensor, index, video_id

    def __len__(self):
        return self.length


def read_dict(filepath):
    f = open(filepath,'r')
    a = f.read()
    dict_data = eval(a)
    f.close()
    return dict_data

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return 1.0 * X / norm



def cosine_sim_np(query_embs, retro_embs):
    query_embs = l2norm(query_embs)
    retro_embs = l2norm(retro_embs)

    return 1.0 - query_embs.dot(retro_embs.T)

def check(resultFile, pattern):
    with open(resultFile) as f:
        datafile = f.readlines()
    for line in datafile:
        if pattern in line:
            print(line.rstrip("\n\r"))


def process(model, device, savefilepath=None, avsyear=None, pre_trained_model='CLIP_RN50', video_set=None):
    batch_size  = 8500

    pre_trained_model = pre_trained_model.replace('-', '_').replace('/','_')

    if avsyear == 'mediaeval25':
        queryfile = '../data/mediaeval/newsimages_25_v1.1/subset.csv'
        df = pd.read_csv(queryfile, sep=",")
        df["article_text"] = ""
        lineList = []
        for idx, row in df.iterrows():
            lineList.append(row["article_title"])
    elif avsyear == 'mediaeval25_summ':
        queryfile = '../data/mediaeval/newsimages_25_v1.1/subset_with_text_summ_capt_fixed.csv'
        df = pd.read_csv(queryfile, sep=",")
        lineList = []
        for idx, row in df.iterrows():
            lineList.append(row["summary"])
    elif avsyear == 'mediaeval25_imgCapti':
        queryfile = '../data/mediaeval/newsimages_25_v1.1/subset_with_text_summ_capt_fixed.csv'
        df = pd.read_csv(queryfile, sep=",")
        lineList = []
        for idx, row in df.iterrows():
            lineList.append(row["caption"])

    # Initialize tokenizer
    tokenizer = SimpleTokenizer()

    tex_feas_all = []
    for line in tqdm.tqdm(lineList):
        try:
            caption = line.strip()
        except:
            caption = 'dummy text'
        # Tokenize and encode text
        text_tokens = tokenizer(caption).cuda(non_blocking=True)
        text_tokens = text_tokens.view(-1, 77).contiguous()
        tex_emb = utils.get_model(model).encode_text(text_tokens)

        tex_feas_all.append(tex_emb.squeeze().detach().cpu().numpy())

    tex_feas_all_np = np.array(tex_feas_all)
    print()

    visual_feat_path = os.path.join('/m2/YFCC100M/YFCC100M_Features/'+ video_set + '/', 'FeatureData', pre_trained_model)
    visual_feats = BigFile( visual_feat_path)
    video2frames = read_dict(os.path.join('/m2/YFCC100M/YFCC100M_Features/'+ video_set + '/', 'FeatureData', pre_trained_model, 'video2frames.txt'))

    dset = Dataset4DualEncoding(visual_feats, 1, video2frames=video2frames)
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               num_workers=5)

    video_ids = []
    errorlistList=[]
    for i, (videos, idxs, vid_ids) in enumerate(tqdm.tqdm(data_loader) ):

        video_ids.extend(vid_ids)

        capt = l2norm(tex_feas_all_np)

        vid = l2norm(videos.squeeze().data.cpu().numpy().copy())
        errors = cosine_sim_np(vid, capt)
        errorlistList.extend(errors)

    print()
    errornp = np.asarray(errorlistList)

    return errornp, video_ids


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [opt] """)
    parser.add_option("--year", default="mediaeval25", type="choice",
                      choices=['mediaeval25', 'mediaeval25_summ', 'mediaeval25_imgCapti'],
                      help="")
    parser.add_option("--video_set", default="YFCC100M", type="choice",
                      choices=['YFCC100M'],
                      help="")
    (opt, args) = parser.parse_args(argv)

    models = ['small', 'base', 'large', 'base_CC3M', 'base_CC12M' ]

    video_set = opt.video_set

    errors = []
    for i, model in enumerate(models):

        if model == 'small':
            ckpt_path = '../models/SLIP/slip_small_100ep.pt'
        elif model == 'base':
            ckpt_path = '../models/SLIP/slip_base_100ep.pt'
        elif model == 'large':
            ckpt_path = '../models/SLIP/slip_large_100ep.pt'
        elif model == 'base_CC3M':
            ckpt_path = '../models/SLIP/slip_base_cc3m_40ep.pt'
        elif model == 'base_CC12M':
            ckpt_path = '../models/SLIP/slip_base_cc12m_35ep.pt'
        else:
            raise Exception('no model found')

        # Load the model.
        device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v

        pre_trained_model = model

        print("creating model: {}")
        print(f"loading checkpoint ' ")
        old_args = ckpt['args']
        print("=> creating model: {}".format(old_args.model))
        model = getattr(SLIP_models, old_args.model)(rand_embed=False,
                                                ssl_mlp_dim=old_args.ssl_mlp_dim, ssl_emb_dim=old_args.ssl_emb_dim)
        model.cuda()
        model.load_state_dict(state_dict, strict=True)
        # print("=> loaded resume checkpoint '{}' (epoch {})".format(args.resume, ckpt['epoch']))


        cudnn.benchmark = True

        avsyear = opt.year
        savefile =  '../data/' + avsyear + '/' + avsyear + '_' + video_set + '_' + 'SLIP_' + pre_trained_model.replace('/','-') + '_similarities.npy'

        error, video_ids = process(model, device, savefile, avsyear, 'SLIP_'+pre_trained_model, video_set)
        with open(savefile, 'wb') as f:
            np.save(f, error)
        errors.append(error)

    errors = np.stack(errors, axis=2)
    sum_array = np.sum(errors, axis=2)
    # Normalize the sum_array
    norm_array = sum_array / np.linalg.norm(sum_array, axis=1, keepdims=True)

    savefile =  '../data/' + avsyear + '/' + avsyear + '_' + video_set + '_' + 'SLIP_similarities.npy'

    with open(savefile, 'wb') as f:
        np.save(f, norm_array)



if __name__ == "__main__":
    sys.exit(main())