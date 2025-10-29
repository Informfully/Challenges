from importlib.metadata import FastPath

import torch.nn as nn
import sys
import time
import os

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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../models")))

import SLIP.SLIP_main.models as models
import SLIP.SLIP_main.utils
import tqdm

def extract_features(model, preprocess, device, imagename):
    image = preprocess(Image.open(imagename)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)

    return image_features


def get_model(model):
    if isinstance(model, torch.nn.DataParallel) \
      or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, keyframe_path_file, val_transform):
        # load your dataset (how every you want, this example has the dataset stored in a json file
        self.Shots = [line.rstrip('\n') for line in open(keyframe_path_file)]
        self.transform = val_transform

    def __getitem__(self, idx):
        sample = self.Shots[idx]
        img = Image.open(sample)
        img_t = self.transform(img)

        srtSplit = sample.split('/')
        keyframe = srtSplit[-1]
        # imageID = keyframe.split('.')[0]
        imageID = os.path.splitext(keyframe)[0]

        return img_t.squeeze(0), imageID

    def __len__(self):
        return len(self.Shots)


def process_batch(model, device, keyframe_path_file=None, savefilepath=None):

    start = time.time()
    # Read AVS_SHOTS
    target = open(savefilepath, 'w')

    c=0

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        lambda x: x.convert('RGB'),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_set = ImageDataset(keyframe_path_file, val_transform)

    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=False,
        num_workers=3, pin_memory=True, drop_last=False)

    with torch.no_grad():
        for images, ids in tqdm.tqdm(train_dataloader):
            # print()
            images = images.cuda(non_blocking=True)

            # encode images
            features_image = get_model(model).encode_image(images)
            # print(features_image.size())
            feas = features_image.cpu().numpy().tolist()

            for fea, id in zip(feas, ids):
                line = id + ' ' + ' '.join(str(item) for item in fea) + '\n'
                target.writelines(line)
                c = c + 1
                if (c % 100000 == 0):
                    end = time.time()
                    # print(str(c) + ' out of ' + str(train_set.__len__()) + ' in: ' + str(end - start))
                    start = time.time()
    target.close()


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [opt] keyframe_path_file feature_file_save_path""")
    parser.add_option("--model", default="slip_small", type="choice",
                      choices=['slip_small', 'slip_base', 'slip_large', 'slip_base_CC3M', 'slip_base_CC12M' ], help="")
    (opt, args) = parser.parse_args(argv)
    if len(args) < 2:
        parser.print_help()
        return 1


    # Load the model.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if opt.model == 'slip_small':
        ckpt_path = '../models/SLIP/pre_trained_models/slip_small_100ep.pt'
    elif opt.model == 'slip_base':
        ckpt_path = '../models/SLIP/pre_trained_models/slip_base_100ep.pt'
    elif opt.model == 'slip_large':
        ckpt_path = '../models/SLIP/pre_trained_models/slip_large_100ep.pt'
    elif opt.model == 'slip_base_CC3M':
        ckpt_path = '../models/SLIP/pre_trained_models/slip_base_cc3m_40ep.pt'
    elif opt.model == 'slip_base_CC12M':
        ckpt_path = '../models/SLIP/pre_trained_models/slip_base_cc12m_35ep.pt'
    else:
        raise Exception('no model found')

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    print("creating model: {}")
    print(f"loading checkpoint ' ")
    old_args = ckpt['args']
    print("=> creating model: {}".format(old_args.model))
    model = getattr(models, old_args.model)(rand_embed=False,
                                            ssl_mlp_dim=old_args.ssl_mlp_dim, ssl_emb_dim=old_args.ssl_emb_dim)
    model.cuda()
    model.load_state_dict(state_dict, strict=True)
    # print("=> loaded resume checkpoint '{}' (epoch {})".format(args.resume, ckpt['epoch']))

    cudnn.benchmark = True

    return process_batch(model, device, args[0], args[1])


if __name__ == "__main__":
    sys.exit(main())