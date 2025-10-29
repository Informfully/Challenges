#!/usr/bin/env python3
"""
OpenClipSelective.py

Train + evaluate a text-to-image retrieval system for MediaEval NewsImages-style data
using OpenCLIP and FAISS. Performs an 80/20 split on news articles, computes retrieval
metrics on the validation split, and saves all artifacts for inference.

Artifacts saved into --out_dir:
  - image_index.faiss         : FAISS index with image embeddings
  - image_embeddings.npy      : (N, D) float32 normalized embeddings
  - image_ids.json            : list of image_ids in the same order as embeddings
  - image_paths.json          : list of resolved file paths (parallel to image_ids)
  - model_config.json         : model + preprocessing config
  - text_tokenizer.pkl        : (not used; OpenCLIP provides tokenizer)
  - eval_metrics.json         : validation metrics (MRR@100, R@k, etc.)

Usage:
  python3 OpenClipSelective.py \
      --csv ../newsimages_25_v1.1/newsarticles.csv \
      --image_dir ../newsimages_25_v1.1/newsimages \
      --out_dir /artifacts \
      --text_cols titleEN textEN \
      --image_id_col image_id \
      --id_col article_id \
      --model_name ViT-H-14 \
      --pretrained laion2b_s32b_b79k \
      --batch_size 64 \
      --num_workers 8 \
      --device cuda

Notes:
- The script expects that the editorially-assigned image for each article exists in `--image_dir`,
  named by its image_id (with common extensions .jpg/.jpeg/.png/.webp). If multiple files match
  (e.g., id.jpg and id.png), the first lexicographic hit is used.
- For evaluation, we embed ALL images from the folder as the candidate pool and compute the
  rank of the ground-truth image for each validation article.
- Dual-softmax re-ranking can be toggled with --dual_softmax.
"""

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
from typing import List, Dict, Tuple, Optional
import pandas as pd
from collections import defaultdict

IMG_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
def find_image_path(image_dir: str, image_id: str) -> Optional[str]:
    """Resolve an image_id to an existing path by trying common extensions."""
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
    text = ". ".join([p for p in parts if p])
    return text

def dual_softmax(Z: np.ndarray) -> np.ndarray:
    """Apply row-wise and column-wise softmax and elementwise multiply (dual softmax)."""
    Z = Z - Z.max(axis=1, keepdims=True)
    row_sm = np.exp(Z)
    row_sm /= (row_sm.sum(axis=1, keepdims=True) + 1e-9)
    Zc = Z - Z.max(axis=0, keepdims=True)
    col_sm = np.exp(Zc)
    col_sm /= (col_sm.sum(axis=0, keepdims=True) + 1e-9)
    return row_sm * col_sm

def recall_at_k(ranks: List[int], k: int) -> float:
    """Recall@k: fraction of queries where the true item is within top-k (rank is 0-based)."""
    hit = sum(1 for r in ranks if r < k)
    return hit / max(1, len(ranks))

def mrr_at_k(ranks: List[int], k: int) -> float:
    """MRR@k: mean reciprocal rank (only counting if rank<k)."""
    rr = [1.0/(r+1) if r < k else 0.0 for r in ranks]
    return float(np.mean(rr)) if rr else 0.0

class ImageFolderDataset(data.Dataset):
    def __init__(self, image_dir: str, image_ids: List[str], preprocess):
        self.image_dir = image_dir
        self.image_ids = image_ids
        self.preprocess = preprocess

        self.paths = []
        for iid in image_ids:
            p = find_image_path(image_dir, iid)
            if p is None:
                self.paths.append(None)
            else:
                self.paths.append(p)

        keep = [(iid,p) for iid,p in zip(self.image_ids, self.paths) if p is not None]
        if len(keep) < len(self.image_ids):
            missing = len(self.image_ids) - len(keep)
            print(f"[warn] {missing} images missing in folder; continuing with {len(keep)} images.", file=sys.stderr)
        self.image_ids = [k[0] for k in keep]
        self.paths = [k[1] for k in keep]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return self.preprocess(img), idx 

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with articles and image_id ground truth")
    ap.add_argument("--image_dir", required=True, help="Folder with candidate images (filenames are image_ids)")
    ap.add_argument("--out_dir", required=True, help="Where to save artifacts")
    ap.add_argument("--text_cols", nargs="+", default=["titleEN","textEN"], help="Text columns to concatenate")
    ap.add_argument("--image_id_col", default="image_id", help="Column with editorial image id")
    ap.add_argument("--id_col", default="article_id", help="Article id column")
    ap.add_argument("--model_name", default="ViT-H-14", help="OpenCLIP model name")
    ap.add_argument("--pretrained", default="laion2b_s32b_b79k", help="OpenCLIP pretrained tag")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dual_softmax", action="store_true", help="Apply dual softmax to similarity matrix before ranking")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    df = pd.read_csv(args.csv)
    for col in [args.image_id_col] + args.text_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in CSV. Found columns: {list(df.columns)}")

    df["__text__"] = df.apply(lambda r: build_text(r, args.text_cols), axis=1)
    df = df[df[args.image_id_col].notnull() & (df["__text__"].str.len() > 0)].copy()
    df[args.image_id_col] = df[args.image_id_col].astype(str)

    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_train = int(0.8 * len(df))
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df   = df.iloc[n_train:].reset_index(drop=True)
    print(f"[info] Total={len(df)} Train={len(train_df)} Val={len(val_df)}")

    print(f"[info] Loading OpenCLIP {args.model_name} ({args.pretrained}) on {args.device}")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained, device=args.device)
    tokenizer = open_clip.get_tokenizer(args.model_name)

    referenced_ids = sorted(set(df[args.image_id_col].astype(str).tolist()))
    folder_ids = []
    for name in os.listdir(args.image_dir):
        base, ext = os.path.splitext(name)
        if ext.lower() in IMG_EXTS:
            folder_ids.append(base)
    all_ids = sorted(set(referenced_ids + folder_ids))
    img_ds = ImageFolderDataset(args.image_dir, all_ids, preprocess)

    model.eval()
    img_loader = data.DataLoader(img_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    all_emb = np.zeros((len(img_ds), model.visual.output_dim), dtype=np.float32)
    with torch.no_grad():
        for batch, idxs in tqdm(img_loader, desc="Encoding images"):
            batch = batch.to(args.device)
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_emb[idxs.numpy()] = feats.cpu().numpy().astype(np.float32)
            
    d = all_emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(all_emb)
    faiss.write_index(index, os.path.join(args.out_dir, "image_index.faiss"))
    np.save(os.path.join(args.out_dir, "image_embeddings.npy"), all_emb)
    with open(os.path.join(args.out_dir, "image_ids.json"), "w") as f:
        json.dump(img_ds.image_ids, f)
    with open(os.path.join(args.out_dir, "image_paths.json"), "w") as f:
        json.dump(img_ds.paths, f)

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
    top5_sims, top5_ids = [], []
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
                D = np.take_along_axis(D, order, axis=1)
            for j in range(len(chunk)):
                gt = val_gt_ids[i+j]
                gt_idx = id_to_idx.get(gt, None)
                if gt_idx is None:
                    continue
                retrieved = I[j].tolist()
                try:
                    rank = retrieved.index(gt_idx)
                except ValueError:
                    rank = 10**9
                ranks.append(rank)
                top5_sims.append(D[j][:5].tolist())
                top5_ids.append([img_ds.image_ids[k] for k in I[j][:5]])

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
    examples = []
    for i in range(min(20, len(top5_ids))):
        examples.append({
            "query_text": val_texts[i],
            "top5_image_ids": top5_ids[i],
            "top5_scores": top5_sims[i]
        })
    with open(os.path.join(args.out_dir, "sample_val_predictions.json"), "w") as f:
        json.dump(examples, f, indent=2)
    print(f"[done] Artifacts saved to: {args.out_dir}")

clip_model = None
clip_preprocess = None
faiss_index = None
image_ids = None
def init_retrieval(model, preprocess, index, ids):
    global clip_model, clip_preprocess, faiss_index, image_ids
    clip_model = model
    clip_preprocess = preprocess
    faiss_index = index
    image_ids = ids

def retrieve_best_image(article_text, model=None, preprocess=None, index=None, ids=None, top_k=1):
    global clip_model, faiss_index, image_ids
    if model is None:
        model = clip_model
    if index is None:
        index = faiss_index
    if ids is None:
        ids = image_ids
    if model is None or index is None or ids is None:
        raise RuntimeError("Retrieval not initialized. Call init_retrieval() or pass model/index/ids explicitly.")

    with torch.no_grad():
        text_tokens = clip.tokenize([article_text]).to(model.visual.weight.device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy().astype(np.float32)
    distances, indices = index.search(text_features, top_k)
    results = [(ids[i], float(distances[0][j])) for j, i in enumerate(indices[0])]
    return results

if __name__ == "__main__":
    main()
