from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

import argparse
import numpy as np
import torch
import os
import sys
from timm.models import create_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../models")))
from beit3.modeling_finetune import beit3_base_patch16_224_retrieval, beit3_base_patch16_384_retrieval, beit3_large_patch16_384_retrieval

import beit3.utils as utils


# Helper function to extract video ID and frame number from filename
def extract_video_id_and_frame(filename):
    parts = os.path.splitext(os.path.basename(filename))[0].split('_')
    return '_'.join(parts[:-1]), parts[-1]  # Adjusting to capture the correct video ID and frame number

# Custom Dataset class
class ImageDataset(Dataset):
    def __init__(self, image_files, transform):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        video_id, frame_number = extract_video_id_and_frame(os.path.basename(image_path))
        # return image, f"{video_id}_{frame_number}"
        return image, f"{frame_number}"

# Function to process images with a given model and save features
def process_and_save_features(dataloader, model, device, output_file):
    # Open the output file at the beginning
    with open(output_file, 'w') as img_f:
        with torch.no_grad():
            for images, frame_ids in tqdm(dataloader, desc=f"Processing images with {output_file}"):
                images = images.to(device)
                vision_cls, _ = model(image=images, only_infer=True)
                vision_cls = vision_cls.cpu().numpy()

                for frame_id, features in zip(frame_ids, vision_cls):
                    feature_str = ' '.join(map(str, features.squeeze()))
                    img_f.write(f"{frame_id} {feature_str}\n")

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
    parser.add_argument('--sentencepiece_model', type=str, required=True,
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
    parser.add_argument('--data_path', default='/m2/YFCC100M/images/', type=str,
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
    parser.add_argument('--keyframe_path_file', default='/m2/YFCC100M/YFCC100M_paths_aa.txt',
                        help='finetune from checkpoint')
    parser.set_defaults(pin_mem=True)

    args = parser.parse_args()

    # Transform for preprocessing images
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
    ])

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


    keyframe_path_file = args.keyframe_path_file
    image_files = [line.rstrip('\n') for line in open(keyframe_path_file)]
    dataset = ImageDataset(image_files, transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Output file name
    output_file = os.path.join(args.output_dir, f"{args.model}_{args.task}_YFCC100M_img_features_mediaeval25.txt")

    # Process and save features
    process_and_save_features(dataloader, model, device, output_file)
