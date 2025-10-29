# Fine-tune CLIP on News Article–Image Pairs (finetune_clip.py)
# Dependencies: torch, torchvision, transformers, pandas, pillow, tqdm

import os
import argparse
import pandas as pd
from PIL import Image
import torch
from torch import nn
torch.backends.cudnn.benchmark = True
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm.auto import tqdm


class NewsImageDataset(Dataset):
    """
    Dataset for news article-image pairs.
    Expects CSV with columns: article_id, article_title, article_tags, image_id
    and images stored under `images_dir/<image_id>.(jpg|png|jpeg)`.
    """
    def __init__(self, csv_path, images_dir, processor, max_length=77):
        self.df = pd.read_csv(csv_path)
        # cast IDs to string
        self.df['article_id'] = self.df['article_id'].astype(str)
        self.df['image_id'] = self.df['image_id'].astype(str)
        self.images_dir = images_dir
        self.processor = processor
        # use CLIP's transform for images
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Text: combine title and tags
        text = f"{row['article_title']} {row.get('article_tags', '')}".strip()
        # Tokenize text
        text_inputs = self.processor(text=[text], return_tensors='pt', padding='max_length', truncation=True)
        # Load image
        img_path = os.path.join(self.images_dir, f"{row['image_id']}.jpg")
        image = Image.open(img_path).convert('RGB')
        image = self.image_transform(image)
        return {
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'pixel_values': image
        }


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    # Load model & processor
    model = CLIPModel.from_pretrained(args.model_name)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model = model.to(device)
    model.train()

    # Dataset & DataLoader
    dataset = NewsImageDataset(args.csv, args.images_dir, processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps
    )

    # Loss: CLIP's inbuilt contrastive loss will be computed by model forward
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                return_loss=True
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        # Save checkpoint
        ckpt_path = os.path.join(args.output_dir, f"clip_finetuned_epoch{epoch+1}.pt")
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Fine-tune CLIP on News Data")
    parser.add_argument('--csv', type=str, default='articles.csv', help='Path to CSV file')
    parser.add_argument('--images_dir', type=str, default='newsimages/', help='Directory of images')
    parser.add_argument('--model_name', type=str, default='openai/clip-vit-base-patch32', help='CLIP model identifier')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Where to save checkpoints')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu','cuda'], help='Compute device')
    args = parser.parse_args()
    train(args)
