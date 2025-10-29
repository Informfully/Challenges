import torch
import os
import numpy as np
import tqdm
import time
import argparse
from basic.bigfile import BigFile
from transformers import XLMRobertaTokenizer
from timm.models import create_model

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import sys
from pathlib import Path
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.utils import ModelEma
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../models")))

from beit3.optim_factory import create_optimizer, get_parameter_groups, \
    LayerDecayValueAssigner, get_is_head_flag_for_vit
from beit3.engine_for_finetuning import train_one_epoch, get_handler, evaluate
from beit3.datasets import create_downstream_dataset
from beit3.utils import NativeScalerWithGradNormCount as NativeScaler
import beit3.utils as utils
import beit3.modeling_finetune
import pandas as pd


def do_L2_norm(vec):
    L2_norm = np.linalg.norm(vec, 2)
    return 1.0 * np.array(vec) / L2_norm


class Dataset4DualEncoding(torch.utils.data.Dataset):
    def __init__(self, visual_feat, do_visual_feas_norm, video2frames=None):

        self.video2frames = video2frames
        self.do_visual_feas_norm = do_visual_feas_norm

        self.video_ids = [key for key in self.video2frames.keys()]
        self.visual_feat = visual_feat

        self.length = len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        frame_list = self.video2frames[video_id]
        frame_vecs = []
        for frame_id in frame_list:
            if self.do_visual_feas_norm:
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
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return 1.0 * X / norm


# Function to process text using BEiT3
def text_encoding(model, tokenizer, text, device):
    tokenized_text = tokenizer(text, return_tensors='pt')
    token_ids = tokenized_text.input_ids.to(device)
    with torch.no_grad():
        _, text_features = model(text_description=token_ids, only_infer=True)
    return text_features.squeeze().cpu().numpy()


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


def process(model, tokenizer, device, savefilepath=None, avsyear=None, pre_trained_model='BEiT3', video_set=None):
    batch_size  = 8500

    if avsyear == 'mediaeval25':
        queryfile = '../data/newsimages_25_v1.1/subset.csv'
        df = pd.read_csv(queryfile, sep=",")
        df["article_text"] = ""
        lineList = []
        for idx, row in df.iterrows():
            lineList.append(row["article_title"])
    elif avsyear == 'mediaeval25_summ':
        queryfile = '../data/newsimages_25_v1.1/subset_with_text_summ_capt.csv'
        df = pd.read_csv(queryfile, sep=",")
        lineList = []
        for idx, row in df.iterrows():
            lineList.append(row["summary"])
    elif avsyear == 'mediaeval25_imgCapti':
        queryfile = '../data/newsimages_25_v1.1/subset_with_text_summ_capt.csv'
        df = pd.read_csv(queryfile, sep=",")
        lineList = []
        for idx, row in df.iterrows():
            lineList.append(row["caption"])


    tex_feas_all = []
    for line in tqdm.tqdm(lineList):
        try:
            caption = line.strip()
        except:
            caption = 'dummy text'
        tex_emb = text_encoding(model, tokenizer, caption, device)

        tex_feas_all.append(tex_emb)

    tex_feas_all_np = np.array(tex_feas_all)
    print()

    if args.task == 'flickr30k':
        # visual_feat_path = os.path.join('/home/aleventakis/Desktop/CERTH_VisualSearch_dualDense/V3C2/', 'FeatureData', pre_trained_model + '_' + args.task)
        visual_feat_path = os.path.join('/m2/YFCC100M/YFCC100M_Features/'+ video_set + '/', 'FeatureData', pre_trained_model + '_' + 'f30k_retrieval')
    else:
        visual_feat_path = os.path.join( '/m2/YFCC100M/YFCC100M_Features/' + video_set + '/', 'FeatureData', pre_trained_model + '_' + args.task)

    visual_feats = BigFile(visual_feat_path)
    video2frames = read_dict(os.path.join(visual_feat_path, 'video2frames.txt'))

    dset = Dataset4DualEncoding(visual_feats, 1, video2frames=video2frames)
    data_loader = torch.utils.data.DataLoader(dataset=dset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                              num_workers=5)

    video_ids = []
    errorlistList = []
    start = time.time()
    for i, (videos, idxs, vid_ids) in enumerate(tqdm.tqdm(data_loader)):
        video_ids.extend(vid_ids)
        capt = l2norm(tex_feas_all_np)
        vid = l2norm(videos.squeeze().data.cpu().numpy().copy())
        errors = cosine_sim_np(vid, capt)
        errorlistList.extend(errors)

    errornp = np.asarray(errorlistList)


    return errornp, video_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images with fine-tuned BEiT3 models')

    parser.add_argument('--model', default='beit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--task', type=str, required=True,
                        choices=['nlvr2', 'vqav2', 'flickr30k', 'coco_retrieval', 'coco_captioning', 'nocaps', 'imagenet'],
                        help='Name of task to fine-tuning')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--checkpoint_activations', action='store_true', default=None,
                        help='Enable checkpointing to save your memory.')
    parser.add_argument('--sentencepiece_model', type=str, default='/home/aleventakis/PycharmProjects/LW_Network/BEIT/unilm-master/beit3/beit3.spm',
                        help='Sentencepiece model path for the pretrained model.')
    parser.add_argument('--vocab_size', type=int, default=64010)
    parser.add_argument('--num_max_bpe_tokens', type=int, default=64)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument("--year", default="mediaeval25_imgCapti", type=str,
                      choices=['mediaeval25', 'mediaeval25_summ', 'mediaeval25_imgCapti'],
                      help="")
    parser.add_argument("--video_set", default="YFCC100M", type=str,
                      choices=['YFCC100M', 'YFCC100M'],
                      help="")

    parser.set_defaults(pin_mem=True)

    args = parser.parse_args()
    # Set device
    device = torch.device(args.device)

    # Set seed for reproducibility
    seed = args.seed + utils.get_rank()
    #seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not args.model.endswith(args.task):
        if args.task in ("flickr30k", "coco_retrieval"):
            model_config = "%s_retrieval" % args.model
        elif args.task in ("coco_captioning", "nocaps"):
            model_config = "%s_captioning" % args.model
        elif args.task in ("imagenet"):
            model_config = "%s_imageclassification" % args.model
        else:
            model_config = "%s_%s" % (args.model, args.task)
    else:
        model_config = args.model
    print("model_config = %s" % model_config)
    model = create_model(
        model_config,
        pretrained=False,
        drop_path_rate=args.drop_path,
        vocab_size=args.vocab_size,
        checkpoint_activations=args.checkpoint_activations,
    )

    if args.finetune:
        utils.load_model_and_may_interpolate(args.finetune, model, args.model_key, args.model_prefix)
    model.to(device)

    errors = []
    tokenizer = XLMRobertaTokenizer("../models/beit3/beit3.spm")

    year = args.year
    video_set = args.video_set
    rephrase = args.rephrase

    savefile = f"../data/{year}/{year}_{args.model.replace('/', '-')}_{args.task}_similarities.npy"


    error, video_ids = process(model, tokenizer, device, savefile, year, args.model, video_set)
    with open(savefile, 'wb') as f:
        np.save(f, error)
    errors.append(error)

    errors = np.stack(errors, axis=2)
    sum_array = np.sum(errors, axis=2)
    norm_array = sum_array / np.linalg.norm(sum_array, axis=1, keepdims=True)

    savefile = f"../data/{year}/{year}_{args.model.replace('/', '-')}_{args.task}_NORM_similarities.npy"

    with open(savefile, 'wb') as f:
        np.save(f, norm_array)
