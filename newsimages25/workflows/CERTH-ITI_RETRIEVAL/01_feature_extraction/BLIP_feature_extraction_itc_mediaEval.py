import tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../models")))

from BLIP.BLIP_main.models.blip_itm_dg import blip_itm_dg

import torch.nn as nn
import sys
import time
import os

from BLIP.BLIP_main.models.blip import blip_feature_extractor
from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def extract_features(model, preprocess, device, imagename):
    image = preprocess(Image.open(imagename)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)

    return image_features


def load_demo_image(sample, image_size, device):
    raw_image = Image.open(sample).convert('RGB')

    w, h = raw_image.size
    # display(raw_image.resize((w // 5, h // 5)))

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0)
    return image

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, keyframe_path_file):
        # load your dataset (how every you want, this example has the dataset stored in a json file
        self.Shots = [line.rstrip('\n') for line in open(keyframe_path_file)]
        # self.preprocess = preprocess

    def __getitem__(self, idx):
        sample = self.Shots[idx]
        image = load_demo_image(sample, 384, 'cuda')
        # image = self.preprocess(Image.open(sample)).unsqueeze(0)

        srtSplit = sample.split('/')
        keyframe = srtSplit[-1]
        # imageID = keyframe.split('.')[0]
        imageID = os.path.splitext(keyframe)[0]

        return image.squeeze(0), imageID

    def __len__(self):
        return len(self.Shots)



def process_batch(model, device, keyframe_path_file=None, savefilepath=None):

    start = time.time()
    # Read AVS_SHOTS
    target = open(savefilepath, 'w')

    c=0

    train_set = ImageDataset(keyframe_path_file)

    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=128,
        num_workers=10,
        shuffle=False,
    )

    for images, ids in tqdm.tqdm(train_dataloader):
        # print()

        with torch.no_grad():
            # image_features = model(images.to(device), 'place holder caption', mode='image')[:,0,:]
            image_features = model(images.to(device), 'place holder caption', match_head='imgfeasnorm')

        if image_features.size()[0] == 1:
            feas =  image_features.cpu().numpy().tolist()
        else:
            feas = image_features.squeeze().cpu().numpy().tolist()

        for fea, id in zip(feas, ids):
            # if id == 'shot17235_9_RKF':
            #     print()
            # if c > 1425400:
            #     print(id)
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
    parser.add_option("--model", default="model_base_retrieval_coco", type="choice", choices=['model_base_retrieval_coco', 'model_base_retrieval_flickr',
                                                                          'model_large_retrieval_coco', 'model_large_retrieval_flickr',], help="")
    (opt, args) = parser.parse_args(argv)
    if len(args) < 2:
        parser.print_help()
        return 1
    # Load the model.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vit_model = 'base'
    image_size = 384
    if 'large' in opt.model:
        vit_model = 'large'


    # model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth'
    model_url = '../models/BLIP/pre_trained_models/' + opt.model + '.pth'
    print(vit_model)
    model = blip_itm_dg(pretrained=model_url, image_size=image_size, vit=vit_model)
    model.eval()
    model = model.to(device)


    return process_batch(model, device, args[0], args[1])


if __name__ == "__main__":
    sys.exit(main())
