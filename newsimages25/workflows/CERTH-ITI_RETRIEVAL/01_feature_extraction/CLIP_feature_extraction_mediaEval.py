import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import sys
import time
import os
import tqdm

import clip


def extract_features(model, preprocess, device, imagename):
    image = preprocess(Image.open(imagename)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)

    return image_features


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, keyframe_path_file, preprocess):
        # load your dataset (how every you want, this example has the dataset stored in a json file
        self.Shots = [line.rstrip('\n') for line in open(keyframe_path_file)]
        self.preprocess = preprocess

    def __getitem__(self, idx):
        sample = self.Shots[idx]
        image = self.preprocess(Image.open(sample)).unsqueeze(0)

        srtSplit = sample.split('/')
        keyframe = srtSplit[-1]
        # imageID = keyframe.split('.')[0]
        imageID = os.path.splitext(keyframe)[0]

        return image.squeeze(0), imageID

    def __len__(self):
        return len(self.Shots)


def process_batch(model, preprocess, device, keyframe_path_file=None, savefilepath=None):

    start = time.time()
    # Read AVS_SHOTS
    target = open(savefilepath, 'w')

    c=0

    train_set = ImageDataset(keyframe_path_file, preprocess)

    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=200,
        num_workers=5,
        shuffle=False,
    )
    for images, ids in tqdm.tqdm(train_dataloader):
        # print()

        with torch.no_grad():
            image_features = model.encode_image(images.to(device))

        feas = image_features.squeeze().cpu().numpy().tolist()

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
    parser.add_option("--model", default="ViT-B/32",  type="choice",
                      choices=['RN50x4', 'RN50x16', 'RN50x64', 'RN50', 'RN101', 'ViT-B/16', 'ViT-B/32', 'ViT-L/14'], help="")
    (opt, args) = parser.parse_args(argv)
    if len(args) < 2:
        parser.print_help()
        return 1
    # Load the model.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pre_trained_model = opt.model
    model, preprocess = clip.load(pre_trained_model, device=device)

    return process_batch(model, preprocess, device, args[0], args[1])


if __name__ == "__main__":
    sys.exit(main())