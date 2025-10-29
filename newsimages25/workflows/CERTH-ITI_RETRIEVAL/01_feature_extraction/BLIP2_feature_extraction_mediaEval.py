
import sys
import time
import os
import tqdm

from lavis.models import load_model_and_preprocess
from PIL import Image
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

    def __init__(self, keyframe_path_file, vis_processors, device):
        # load your dataset (how every you want, this example has the dataset stored in a json file
        self.Shots = [line.rstrip('\n') for line in open(keyframe_path_file)]
        self.vis_processors = vis_processors
        self.device = device

    def __getitem__(self, idx):
        sample = self.Shots[idx]
        raw_image = Image.open(sample).convert('RGB')
        image = self.vis_processors["eval"](raw_image).unsqueeze(0)
        # image = load_demo_image(sample, 224, 'cuda')
        # image = self.preprocess(Image.open(sample)).unsqueeze(0)

        srtSplit = sample.split('/')
        keyframe = srtSplit[-1]
        # imageID = keyframe.split('.')[0]
        imageID = os.path.splitext(keyframe)[0]

        return image.squeeze(0), imageID

    def __len__(self):
        return len(self.Shots)


def process_batch(model, vis_processors, txt_processors, device, keyframe_path_file=None, savefilepath=None):

    # Read AVS_SHOTS
    target = open(savefilepath, 'w')

    train_set = ImageDataset(keyframe_path_file, vis_processors, device)

    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=128,
        num_workers=5,
        shuffle=False,
    )

    for images, ids in tqdm.tqdm(train_dataloader):
        sample = {"image": images.to(device)}
        with torch.no_grad():
            features_image = model.extract_features(sample, mode="image")
        feas = features_image.image_embeds_proj[:,0,:].cpu().numpy().tolist()
        for fea, id in zip(feas, ids):
            line = id + ' ' + ' '.join(str(item) for item in fea) + '\n'
            target.writelines(line)
    target.close()


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [opt] keyframe_path_file feature_file_save_path""")
    parser.add_option("--model", default="coco", type="choice",
                      choices=['pretrain_vitL', 'pretrain', 'coco' ], help="")
    (opt, args) = parser.parse_args(argv)
    if len(args) < 2:
        parser.print_help()
        return 1
    # Load the model.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model_type = “pretrain”, “pretrain_vitL” or “coco”
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_image_text_matching",
                                                                      model_type=opt.model, is_eval=True,
                                                                      device=device)

    return process_batch(model,  vis_processors, txt_processors, device, args[0], args[1])


if __name__ == "__main__":
    sys.exit(main())