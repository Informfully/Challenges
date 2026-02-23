# News Image Retrieval Pipeline (pipeline.py)
# Dependencies: torch, transformers, faiss-cpu, pandas, pillow
#
# Usage:
# 1. Build index:
#    python pipeline.py --mode build --csv articles.csv --images_dir newsimages/ --index_path index.faiss --ids_path ids.npy
#
# 2. Retrieve images by article ID:
#    python pipeline.py --mode retrieve --article_id <ARTICLE_ID> --csv articles.csv --index_path index.faiss --ids_path ids.npy --k 5
#
# 3. Retrieve images by custom text:
#    python pipeline.py --mode retrieve --text "Breaking news about climate change" --index_path index.faiss --ids_path ids.npy --k 5

import os
import argparse
import numpy as np
import pandas as pd
import faiss
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple


def load_articles(csv_path: str) -> pd.DataFrame:
    """Load articles CSV into a DataFrame and ensure article_id is string."""
    df = pd.read_csv(csv_path)
    # Cast article_id to str for consistent lookups
    if 'article_id' in df.columns:
        df['article_id'] = df['article_id'].astype(str)
    return df


def load_image_paths(images_dir: str) -> dict:
    """Map image_id to local file path."""
    image_paths = {}
    for fname in os.listdir(images_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_id = os.path.splitext(fname)[0]
            image_paths[image_id] = os.path.join(images_dir, fname)
    return image_paths


def build_image_embeddings(
    image_paths: dict,
    processor: CLIPProcessor,
    model: CLIPModel,
    device: str = 'cpu'
) -> Tuple[np.ndarray, List[str]]:
    """Compute normalized CLIP image embeddings for all images."""
    model.to(device)
    embeddings, ids = [], []
    with torch.no_grad():
        for idx, (image_id, path) in enumerate(image_paths.items()):
            img = Image.open(path).convert('RGB')
            inputs = processor(images=img, return_tensors='pt').to(device)
            emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy())
            ids.append(image_id)
            if idx and idx % 1000 == 0:
                print(f"Processed {idx}/{len(image_paths)} images")
    emb_matrix = np.vstack(embeddings).astype('float32')
    return emb_matrix, ids


def build_faiss_index(
    emb_matrix: np.ndarray,
    n_neighbors: int = 32
) -> faiss.Index:
    """Create an HNSW index over the embedding matrix."""
    d = emb_matrix.shape[1]
    index = faiss.IndexHNSWFlat(d, n_neighbors)
    index.hnsw.efConstruction = 40
    index.add(emb_matrix)
    return index


def save_index(
    index: faiss.Index,
    ids: List[str],
    index_path: str,
    ids_path: str
):
    """Persist FAISS index and ID mapping to disk."""
    faiss.write_index(index, index_path)
    np.save(ids_path, np.array(ids))


def load_faiss_index(
    index_path: str,
    ids_path: str
) -> Tuple[faiss.Index, List[str]]:
    """Load FAISS index and ID mapping from disk."""
    index = faiss.read_index(index_path)
    ids = np.load(ids_path).tolist()
    return index, ids


def embed_text(
    text: str,
    processor: CLIPProcessor,
    model: CLIPModel,
    device: str = 'cpu'
) -> np.ndarray:
    """Compute normalized CLIP text embedding."""
    model.to(device)
    inputs = processor(text=[text], return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype('float32')


def retrieve_images(
    query: str,
    processor: CLIPProcessor,
    model: CLIPModel,
    index: faiss.Index,
    ids: List[str],
    k: int = 5,
    device: str = 'cpu'
) -> List[Tuple[str, float]]:
    """Return top-k (image_id, score) pairs for a given text query."""
    text_emb = embed_text(query, processor, model, device)
    distances, indices = index.search(text_emb, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append((ids[idx], float(dist)))
    return results


def main():
    parser = argparse.ArgumentParser("News Image Retrieval Pipeline")
    parser.add_argument('--mode', choices=['build', 'retrieve'], required=True,
                        help="Mode 'build' to create index, 'retrieve' for querying images.")
    parser.add_argument('--csv', type=str, default='articles.csv',
                        help="Path to articles CSV")
    parser.add_argument('--images_dir', type=str, default='newsimages/',
                        help="Directory of image files")
    parser.add_argument('--index_path', type=str, default='index.faiss',
                        help="Path to save/load FAISS index")
    parser.add_argument('--ids_path', type=str, default='ids.npy',
                        help="Path to save/load image ID list")
    parser.add_argument('--device', type=str, default='cpu',
                        help="Computation device ('cpu' or 'cuda')")
    parser.add_argument('--text', type=str,
                        help="Custom query text for retrieval mode")
    parser.add_argument('--article_id', type=str,
                        help="Article ID to query for retrieval mode")
    parser.add_argument('--k', type=int, default=5,
                        help="Number of nearest neighbors to return")
    args = parser.parse_args()

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    if args.mode == 'build':
        df = load_articles(args.csv)
        image_paths = load_image_paths(args.images_dir)
        emb_matrix, ids = build_image_embeddings(image_paths, processor, model, args.device)
        index = build_faiss_index(emb_matrix)
        save_index(index, ids, args.index_path, args.ids_path)
        print("Index built and saved.")

    else:  # retrieve
        index, ids = load_faiss_index(args.index_path, args.ids_path)
        df = load_articles(args.csv)
        if args.article_id:
            # ensure argument is string
            art_id = str(args.article_id)
            row = df[df['article_id'] == art_id]
            if row.empty:
                print(f"Article ID {art_id} not found.")
                return
            title = row.iloc[0]['article_title']
            tags = row.iloc[0].get('article_tags', '')
            query = f"{title} {tags}".strip()
        elif args.text:
            query = args.text
        else:
            print("Provide --text or --article_id for retrieval mode.")
            return

        results = retrieve_images(query, processor, model, index, ids, args.k, args.device)
        print(f"Query: {query}\nResults:")
        for img_id, score in results:
            print(f"  Image ID: {img_id}, Score: {score}")


if __name__ == "__main__":
    main()