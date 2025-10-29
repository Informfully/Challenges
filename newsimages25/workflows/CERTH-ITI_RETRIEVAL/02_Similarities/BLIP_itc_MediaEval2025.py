import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../models")))

from BLIP.BLIP_main.models.blip import blip_feature_extractor
from BLIP.BLIP_main.models.blip_itm_dg import blip_itm_dg
import torch
from PIL import Image

import numpy as np
import requests
from torchvision import transforms

import tqdm
import time
# from scipy.special import softmax

from basic.bigfile import BigFile
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import pandas as pd


def do_L2_norm(vec):
    L2_norm = np.linalg.norm(vec, 2)
    return 1.0 * np.array(vec) / L2_norm


class Dataset4DualEncoding(torch.utils.data.Dataset):
    """
       Load captions and video frame features by pre-trained CNN model.
       """

    def __init__(self, visual_feat, do_visual_feas_norm, video2frames=None):

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
    f = open(filepath, 'r')
    a = f.read()
    dict_data = eval(a)
    f.close()
    return dict_data


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return 1.0 * X / norm


def load_demo_image(image_size, device):
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    w, h = raw_image.size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


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
    batch_size = 8500

    pre_trained_model = pre_trained_model.replace('-', '_').replace('/', '_')

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
    image = load_demo_image(image_size=224, device=device)
    for line in tqdm.tqdm(lineList):
        try:
            caption = line.strip()
        except:
            caption = 'dummy text'
        with torch.no_grad():
            tex_emb = model(image, caption, match_head='texteasnorm')
            # a = model.text_encoder(caption)
        # tex_emb = text_encoding(model, preprocess, caption, device)
        tex_feas_all.append(tex_emb.squeeze().cpu().numpy())

    tex_feas_all_np = np.array(tex_feas_all)
    print()

    visual_feat_path = os.path.join('/m2/YFCC100M/YFCC100M_Features/' + video_set + '/', 'FeatureData',
                                    pre_trained_model)
    visual_feats = BigFile(visual_feat_path)
    video2frames = read_dict(
        os.path.join('/m2/YFCC100M/YFCC100M_Features/' + video_set + '/', 'FeatureData', pre_trained_model,
                     'video2frames.txt'))

    dset = Dataset4DualEncoding(visual_feats, 1, video2frames=video2frames)
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=5)

    video_ids = []
    errorlistList = []
    start = time.time()
    for i, (videos, idxs, vid_ids) in enumerate(tqdm.tqdm(data_loader)):
        video_ids.extend(vid_ids)

        capt = l2norm(tex_feas_all_np)
        # a = torch.from_numpy(tex_feas_all_np)
        # capt2 = a / a.norm(dim=-1, keepdim=True, p=2)

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

    models = ['model_base_retrieval_coco', 'model_base_retrieval_flickr', 'model_large_retrieval_coco',
              'model_large_retrieval_flickr']

    video_set = opt.video_set

    errors = []
    for i, model in enumerate(models):

        # Load the model.
        vit_model = 'base'
        if 'large' in model:
            vit_model = 'large'
        device = "cuda" if torch.cuda.is_available() else "cpu"

        pre_trained_model = model

        model_url = '../models/BLIP/pre_trained_models/' + pre_trained_model + '.pth'
        print(vit_model)
        model = blip_itm_dg(pretrained=model_url, image_size=224, vit=vit_model)
        model.eval()
        model = model.to(device)

        year = opt.year
        savefile = f"../data/{year}/{year}_{video_set}_BLIP_itc_{pre_trained_model.replace('/', '-')}_similarities.npy"

        error, video_ids = process(model, device, savefile, year, 'BLIP_itc_' + pre_trained_model, video_set)
        with open(savefile, 'wb') as f:
            np.save(f, error)
        errors.append(error)

    errors = np.stack(errors, axis=2)
    sum_array = np.sum(errors, axis=2)
    # Normalize the sum_array
    norm_array = sum_array / np.linalg.norm(sum_array, axis=1, keepdims=True)
    savefile = f'../data/{year}/{year}_{video_set}BLIP_itc_similarities.npy'

    with open(savefile, 'wb') as f:
        np.save(f, norm_array)


if __name__ == "__main__":
    sys.exit(main())
