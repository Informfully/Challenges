import os
import json
import csv
import faiss
import numpy as np
import pandas as pd
from PIL import Image
import zipfile
import shutil
import torch
import open_clip
from tqdm import tqdm

FAISS_INDEX_PATH = "./artifacts/image_index.faiss"
IMAGE_EMBEDDINGS_PATH = "./artifacts/image_embeddings.npy"
IMAGE_IDS_JSON = "./artifacts/image_ids.json"
IMAGE_PATHS_JSON = "./artifacts/image_paths.json"
ARTICLES_CSV = "../newsimages_25_v1.1/newsarticles.csv"
SUBSET_CSV = "../newsimages_25_v1.1/subset.csv"
OUTPUT_MAPPING_CSV = "./article_best_match.csv"
OUTPUT_ZIP = "./Headline_Hunters.zip"

GROUP_NAME = "Headline_Hunters"
APPROACH_NAME = "OpenClipSelective"
PREFIX_RET = "RET"

LARGE_DIRNAME = f"{PREFIX_RET}_{APPROACH_NAME}_LARGE"
SMALL_DIRNAME = f"{PREFIX_RET}_{APPROACH_NAME}_SMALL"

TARGET_W, TARGET_H = 460, 260
TARGET_FORMAT = "PNG"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
DEVICE = torch.device("cpu")

def load_json_list(path):
    with open(path, "r") as f:
        return json.load(f)

def safe_makedirs(path):
    os.makedirs(path, exist_ok=True)

def load_faiss_index(path):
    print(f"Loading FAISS index from: {path}")
    index = faiss.read_index(path)
    print(f"FAISS index dimension: {index.d}")
    return index

def l2_normalize_numpy(x, eps=1e-10):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms

def encode_texts_openclip(model, texts, device):
    batch_size = 64
    all_feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
        batch = texts[i : i + batch_size]
        tokens = open_clip.tokenize(batch).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
        feats = feats.cpu().numpy()
        all_feats.append(feats)
    feats = np.vstack(all_feats).astype(np.float32)
    feats = l2_normalize_numpy(feats)
    return feats

def get_image_path_for_id(image_id, id_to_path_dict):
    p = id_to_path_dict.get(image_id)
    if p and os.path.exists(p):
        return p
    if p:
        base, _ = os.path.splitext(p)
        for ext_try in [".jpg", ".jpeg", ".png"]:
            candidate = base + ext_try
            if os.path.exists(candidate):
                return candidate
    for candidate in id_to_path_dict.values():
        if image_id in os.path.basename(candidate) and os.path.exists(candidate):
            return candidate
    return None

def resize_and_save_png(src_path, dst_path, width=460, height=260):
    im = Image.open(src_path).convert("RGB")
    im = im.resize((width, height), Image.LANCZOS)
    safe_makedirs(os.path.dirname(dst_path))
    im.save(dst_path, format=TARGET_FORMAT)

def main():
    print("Loading image ids and paths...")
    image_ids = load_json_list(IMAGE_IDS_JSON)
    image_paths = load_json_list(IMAGE_PATHS_JSON)
    id_to_path = {iid: p for iid, p in zip(image_ids, image_paths)}

    print("Loading articles from:", ARTICLES_CSV)
    articles = pd.read_csv(ARTICLES_CSV, dtype=str)
    id_col, title_col, tags_col = "article_id", "article_title", "article_tags"
    for c in [id_col, title_col, tags_col]:
        if c not in articles.columns:
            raise ValueError(f"Expected column '{c}' in {ARTICLES_CSV}")

    articles["query_text"] = (articles[title_col].fillna("") + " " + articles[tags_col].fillna("")).str.strip()
    article_ids = articles[id_col].tolist()
    query_texts = articles["query_text"].tolist()

    print(f"Loading OpenCLIP model {MODEL_NAME}, pretrained={PRETRAINED}")
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model.to(DEVICE).eval()

    text_embeddings = encode_texts_openclip(model, query_texts, DEVICE)
    print("Encoded embeddings shape:", text_embeddings.shape)
    index = load_faiss_index(FAISS_INDEX_PATH)

    if index.d != text_embeddings.shape[1]:
        raise ValueError(f"Dimension mismatch: FAISS index dim {index.d}, text embeddings dim {text_embeddings.shape[1]}")

    print("Searching FAISS index...")
    k = 1
    queries = np.ascontiguousarray(text_embeddings.astype(np.float32))
    distances, indices = index.search(queries, k)

    best_image_ids = []
    for idx_row in indices:
        idx = int(idx_row[0])
        best_image_ids.append(image_ids[idx] if 0 <= idx < len(image_ids) else "")
    print("Writing mapping CSV to:", OUTPUT_MAPPING_CSV)
    with open(OUTPUT_MAPPING_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["article_id", "best_image_id"])
        writer.writerows(zip(article_ids, best_image_ids))

    tmp_root = "/tmp/headline_hunters_submission"
    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)
    large_dir = os.path.join(tmp_root, LARGE_DIRNAME)
    small_dir = os.path.join(tmp_root, SMALL_DIRNAME)
    safe_makedirs(large_dir)
    safe_makedirs(small_dir)
    mapping = dict(zip(article_ids, best_image_ids))

    subset_ids = []
    if os.path.exists(SUBSET_CSV):
        df_sub = pd.read_csv(SUBSET_CSV, header=None, dtype=str)
        subset_ids = df_sub.iloc[:, 0].astype(str).tolist()

    def dst_png_path(out_dir, article_id):
        return os.path.join(out_dir, f"{article_id}_{GROUP_NAME}_{APPROACH_NAME}.png")

    def process_article_list(article_list, out_dir):
        n_ok, n_missing = 0, 0
        for aid in tqdm(article_list, desc=f"Processing {os.path.basename(out_dir)}"):
            best_img_id = mapping.get(str(aid), "")
            if not best_img_id:
                n_missing += 1
                continue
            src_path = get_image_path_for_id(best_img_id, id_to_path)
            if not src_path:
                n_missing += 1
                continue
            dst_path = dst_png_path(out_dir, str(aid))
            try:
                resize_and_save_png(src_path, dst_path, TARGET_W, TARGET_H)
                n_ok += 1
            except Exception as e:
                print(f"Error processing {src_path} for article {aid}: {e}")
                n_missing += 1
        return n_ok, n_missing

    print("Preparing LARGE submission...")
    n_ok_large, n_missing_large = process_article_list(article_ids, large_dir)

    print("Preparing SMALL submission...")
    small_ids = [aid for aid in subset_ids if str(aid) in mapping]
    n_ok_small, n_missing_small = process_article_list(small_ids, small_dir)

    print("Creating ZIP:", OUTPUT_ZIP)
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(large_dir):
            for fn in files:
                full = os.path.join(root, fn)
                arcname = os.path.join(GROUP_NAME, LARGE_DIRNAME, fn)
                zf.write(full, arcname)
        for root, _, files in os.walk(small_dir):
            for fn in files:
                full = os.path.join(root, fn)
                arcname = os.path.join(GROUP_NAME, SMALL_DIRNAME, fn)
                zf.write(full, arcname)

    print("Done.")
    print(" - Mapping CSV:", OUTPUT_MAPPING_CSV)
    print(" - Submission ZIP:", OUTPUT_ZIP)
    print(f"LARGE images: {n_ok_large} (missing {n_missing_large})")
    print(f"SMALL images: {n_ok_small} (missing {n_missing_small})")

if __name__ == "__main__":
    main()
