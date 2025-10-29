#!/usr/bin/env python3
import argparse
import os
import sys
import json
import random
import numpy as np
import faiss
from tqdm import tqdm
from PIL import Image
import torch
import torch.utils.data as data
import open_clip
from typing import List, Optional
import pandas as pd

IMG_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

def find_image_path(image_dir: str, image_id: str) -> Optional[str]:
    base = os.path.join(image_dir, str(image_id))
    for ext in IMG_EXTS:
        p = base + ext
        if os.path.exists(p):
            return p
    if os.path.exists(os.path.join(image_dir, image_id)):
        return os.path.join(image_dir, image_id)
    return None

def build_text(series: pd.Series, text_cols: List[str]) -> str:
    parts = []
    for c in text_cols:
        if c in series and pd.notnull(series[c]):
            parts.append(str(series[c]).strip())
    return ". ".join([p for p in parts if p])

def dual_softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(axis=1, keepdims=True)
    row_sm = np.exp(Z)
    row_sm /= (row_sm.sum(axis=1, keepdims=True) + 1e-9)
    Zc = Z - Z.max(axis=0, keepdims=True)
    col_sm = np.exp(Zc)
    col_sm /= (col_sm.sum(axis=0, keepdims=True) + 1e-9)
    return row_sm * col_sm

def recall_at_k(ranks: List[int], k: int) -> float:
    hit = sum(1 for r in ranks if r < k)
    return hit / max(1, len(ranks))

def mrr_at_k(ranks: List[int], k: int) -> float:
    rr = [1.0/(r+1) if r < k else 0.0 for r in ranks]
    return float(np.mean(rr)) if rr else 0.0

class ImageFolderDataset(data.Dataset):
    def __init__(self, image_dir: str, preprocess):
        self.image_dir = image_dir
        self.preprocess = preprocess
        self.paths = []
        self.image_ids = []
        for name in os.listdir(image_dir):
            base, ext = os.path.splitext(name)
            if ext.lower() in IMG_EXTS:
                self.paths.append(os.path.join(image_dir, name))
                self.image_ids.append(base)

        if len(self.paths) == 0:
            raise ValueError(f"No valid images found in {image_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return self.preprocess(img), idx
        
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with articles and image_id ground truth")
    ap.add_argument("--image_dir", required=True, help="Folder with candidate images")
    ap.add_argument("--out_dir", required=True, help="Where to save artifacts")
    ap.add_argument("--text_cols", nargs="+", default=None, help="Text columns to concatenate")
    ap.add_argument("--image_id_col", default="image_id")
    ap.add_argument("--id_col", default="article_id")
    ap.add_argument("--model_name", default="ViT-H/14")
    ap.add_argument("--pretrained", default="laion2b_s32b_b79k")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    try:
        if torch.backends.mps.is_available():
            device = "mps"
            _ = torch.tensor([1.0], device=device) * 2
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    except Exception:
        device = "cpu"
    ap.add_argument("--device", default=device)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dual_softmax", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    df = pd.read_csv(args.csv)
    if not args.text_cols:
        available = set(df.columns)
        if "article_title" in available and "article_tags" in available:
            args.text_cols = ["article_title", "article_tags"]
            print(f"[info] Auto-using text columns: {args.text_cols}")
        elif "article_title" in available:
            args.text_cols = ["article_title"]
            print(f"[info] Auto-using text column: {args.text_cols}")
        elif "titleEN" in available and "textEN" in available:
            args.text_cols = ["titleEN", "textEN"]
            print(f"[info] Auto-using text columns: {args.text_cols}")
        else:
            raise ValueError(f"No suitable text columns found. Available: {list(df.columns)}")

    for col in [args.image_id_col] + args.text_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in CSV. Found: {list(df.columns)}")

    # Create text field
    df["__text__"] = df.apply(lambda r: build_text(r, args.text_cols), axis=1)
    df = df[df[args.image_id_col].notnull() & (df["__text__"].str.len() > 0)].copy()
    df[args.image_id_col] = df[args.image_id_col].astype(str)

    # Split
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_train = int(0.8 * len(df))
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df   = df.iloc[n_train:].reset_index(drop=True)
    print(f"[info] Total={len(df)} Train={len(train_df)} Val={len(val_df)}")

    # Load OpenCLIP
    print(f"[info] Loading OpenCLIP {args.model_name} ({args.pretrained}) on {args.device}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained, device=args.device
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()

    # Image dataset (ALL images in folder)
    img_ds = ImageFolderDataset(args.image_dir, preprocess)

    # Encode images
    img_loader = data.DataLoader(img_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    all_emb = np.zeros((len(img_ds), model.visual.output_dim), dtype=np.float32)
    with torch.no_grad():
        for batch, idxs in tqdm(img_loader, desc="Encoding images"):
            batch = batch.to(args.device)
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_emb[idxs.numpy()] = feats.cpu().numpy().astype(np.float32)

    # Build FAISS index
    d = all_emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(all_emb)
    faiss.write_index(index, os.path.join(args.out_dir, "image_index.faiss"))
    np.save(os.path.join(args.out_dir, "image_embeddings.npy"), all_emb)
    with open(os.path.join(args.out_dir, "image_ids.json"), "w") as f:
        json.dump(img_ds.image_ids, f)
    with open(os.path.join(args.out_dir, "image_paths.json"), "w") as f:
        json.dump(img_ds.paths, f)

    # Save config
    config = dict(
        model_name=args.model_name,
        pretrained=args.pretrained,
        text_cols=args.text_cols,
        image_id_col=args.image_id_col,
        id_col=args.id_col,
        device=args.device,
        dual_softmax=args.dual_softmax,
    )
    with open(os.path.join(args.out_dir, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    val_texts = val_df["__text__"].tolist()
    val_gt_ids = val_df[args.image_id_col].astype(str).tolist()
    id_to_idx = {iid: i for i, iid in enumerate(img_ds.image_ids)}

    ranks = []
    B = args.batch_size
    with torch.no_grad():
        for i in tqdm(range(0, len(val_texts), B), desc="Evaluating"):
            chunk = val_texts[i:i+B]
            toks = tokenizer(chunk).to(args.device)
            tfeat = model.encode_text(toks)
            tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
            t_np = tfeat.cpu().numpy().astype(np.float32)

            D, I = index.search(t_np, 100)
            if args.dual_softmax:
                Z = D.copy()
                Z_ds = dual_softmax(Z)
                order = np.argsort(-Z_ds, axis=1)
                I = np.take_along_axis(I, order, axis=1)

            for j in range(len(chunk)):
                gt = val_gt_ids[i+j]
                gt_idx = id_to_idx.get(gt, None)
                if gt_idx is None: continue
                retrieved = I[j].tolist()
                try:
                    rank = retrieved.index(gt_idx)
                except ValueError:
                    rank = 10**9
                ranks.append(rank)

    metrics = {
        "MRR@100": mrr_at_k(ranks, 100),
        "Recall@5": recall_at_k(ranks, 5),
        "Recall@10": recall_at_k(ranks, 10),
        "Recall@50": recall_at_k(ranks, 50),
        "Recall@100": recall_at_k(ranks, 100),
        "num_evaluated": len(ranks)
    }
    print("[metrics]", json.dumps(metrics, indent=2))
    with open(os.path.join(args.out_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[done] Artifacts saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
