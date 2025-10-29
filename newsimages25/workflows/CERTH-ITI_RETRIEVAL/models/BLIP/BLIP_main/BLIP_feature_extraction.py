
import torch.nn as nn
import sys
import time
import os

from BLIP_main.models.blip import blip_feature_extractor
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
        image = load_demo_image(sample, 224, 'cuda')
        # image = self.preprocess(Image.open(sample)).unsqueeze(0)

        srtSplit = sample.split('/')
        keyframe = srtSplit[-1]
        # imageID = keyframe.split('.')[0]
        imageID = os.path.splitext(keyframe)[0]

        return image.squeeze(0), imageID

    def __len__(self):
        return len(self.Shots)


def process(model, preprocess, device, keyframe_path_file=None, savefilepath=None):

    start = time.time()
    # Read AVS_SHOTS
    Shots = [line.rstrip('\n') for line in open(keyframe_path_file)]
    target = open(savefilepath, 'w')

    imagesSets=[]
    c=0
    for img_path in Shots:
        srtSplit = img_path.split('/')
        videoID = srtSplit[-2]
        keyframe = srtSplit[-1]

        imageID = keyframe.split('.')[0]

        features = extract_features(model, preprocess, device, img_path )

        line = imageID + ' ' + ' '.join(str(item) for item in features.squeeze().numpy().tolist()) + '\n'
        target.writelines(line)
        c = c + 1
        if (c % 1000 == 0):
            end = time.time()
            print(str(c) + ' out of ' + str(Shots.__len__()) + ' in: ' + str(end - start))
            start = time.time()
    target.close()


def process_batch(model, device, keyframe_path_file=None, savefilepath=None):

    start = time.time()
    # Read AVS_SHOTS
    target = open(savefilepath, 'w')

    c=0

    train_set = ImageDataset(keyframe_path_file)

    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=50,
        num_workers=3,
        shuffle=False,
    )

    for images, ids in train_dataloader:
        # print()

        with torch.no_grad():
            image_features = model(images.to(device), 'place holder caption', mode='image')[:,0,:]
        feas = image_features.squeeze().cpu().numpy().tolist()

        for fea, id in zip(feas, ids):
            line = id + ' ' + ' '.join(str(item) for item in fea) + '\n'
            target.writelines(line)
            c = c + 1
            if (c % 1000 == 0):
                end = time.time()
                print(str(c) + ' out of ' + str(train_set.__len__()) + ' in: ' + str(end - start))
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
    if 'large' in opt.model:
        vit_model = 'large'

    # model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth'
    model_url = '/home/dgalanop/PycharmProjects/AVS_2024/_00_2023_reproduction/BLIP/pre_trained_models/' + opt.model + '.pth'
    
    model = blip_feature_extractor(pretrained=model_url, image_size=224, vit=vit_model)
    model.eval()
    model = model.to(device)


    return process_batch(model, device, args[0], args[1])


if __name__ == "__main__":
    sys.exit(main())