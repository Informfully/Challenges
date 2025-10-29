import torch
import os
import argparse
from collections import OrderedDict
import models
from tokenizer import SimpleTokenizer
import utils

def get_args_parser():
    parser = argparse.ArgumentParser(description='SLIP 0-shot evaluations', add_help=False)
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('--batch-size', default=256, type=int, help='batch_size')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
    return parser


def load_titles_and_ids_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]

    titles_and_ids = [line.split('#enc#0') for line in lines]
    image_ids = [parts[0].strip() for parts in titles_and_ids]
    titles = [parts[1].strip() if len(parts) > 1 else '' for parts in titles_and_ids]

    return image_ids, titles

"""
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        lambda x: x.convert('RGB'),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    return transform(image)
"""

def main(args):

    choice = 'S'
    model_path = '/home/aleventakis/PycharmProjects/SLIP_embeddings/slip_small_100ep.pt'

    if choice == 'S':
        model_path = '/home/aleventakis/PycharmProjects/SLIP_embeddings/slip_small_100ep.pt'
    elif choice == 'B':
        model_path = '/home/aleventakis/PycharmProjects/SLIP_embeddings/slip_base_100ep.pt'
    elif choice == 'B_CC3M':
        model_path = 'path_to_B_CC3M_checkpoint'
    elif choice == 'B_CC12M':
        model_path = 'path_to_B_CC12M_checkpoint'
    elif choice == 'L':
        model_path = 'path_to_L_checkpoint'

    if args.resume:
        ckpt_path = args.resume
    elif os.path.isfile(os.path.join(args.output_dir, model_path)):
        ckpt_path = os.path.join(args.output_dir, model_path)
    else:
        raise Exception('no checkpoint found')

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    # create model
    old_args = ckpt['args']
    print("=> creating model: {}".format(old_args.model))
    model = getattr(models, old_args.model)(rand_embed=False,
                                            ssl_mlp_dim=old_args.ssl_mlp_dim, ssl_emb_dim=old_args.ssl_emb_dim)
    model.cuda()
    model.load_state_dict(state_dict, strict=True)
    print("=> loaded resume checkpoint '{}' (epoch {})".format(args.resume, ckpt['epoch']))

    model.eval()

    # Initialize tokenizer
    tokenizer = SimpleTokenizer()

    output_dir = '/home/aleventakis/PycharmProjects/SLIP_embeddings/SLIP_TXT_EMB'

    # Load image IDs and titles from file
    image_ids, titles = load_titles_and_ids_from_file(
        '/home/aleventakis/Desktop/Datasets/05.CNN_news/training_txts/cnn_title.txt')  # Replace with the path to your text file

    for image_id, title in zip(image_ids, titles):

        # Tokenize and encode text
        text_tokens = tokenizer(title).cuda(non_blocking=True)
        text_tokens = text_tokens.view(-1, 77).contiguous()
        text_embedding = utils.get_model(model).encode_text(text_tokens)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        #text_embedding = text_embedding.mean(dim=0)
        #text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        # Save the text embedding to a file with image ID in the name
        output_file = os.path.join(output_dir, f'{image_id}_embedding.txt')
        with open(output_file, 'w') as output:
            output.write(','.join(map(str, text_embedding.tolist())))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('SLIP 0-shot evaluations', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)