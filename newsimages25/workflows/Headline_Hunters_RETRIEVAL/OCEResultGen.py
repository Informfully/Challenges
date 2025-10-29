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

ARTIFACTS_DIR = "./artifacts"
CSV_PATH = "../newsimages_25_v1.1/newsarticles.csv"
SUBSET_CSV = "../newsimages_25_v1.1/subset.csv"
OUT_DIR = "./"

GROUP_NAME = "Headline_Hunters"
APPROACH_NAME = "OpenClipSelective"
PREFIX_RET = "RET"

LARGE_DIRNAME = f"{PREFIX_RET}_{APPROACH_NAME}_LARGE"
SMALL_DIRNAME = f"{PREFIX_RET}_{APPROACH_NAME}_SMALL"

TARGET_W, TARGET_H = 460, 260
TARGET_FORMAT = "PNG"
DEVICE = "cpu"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def safe_makedirs(path):
    os.makedirs(path, exist_ok=True)

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

def resize_and_save_png(src_path, dst_path, width=460, height=260):
    im = Image.open(src_path).convert("RGB")
    im = im.resize((width, height), Image.LANCZOS)
    safe_makedirs(os.path.dirname(dst_path))
    im.save(dst_path, format=TARGET_FORMAT)

def main():
    faiss_index_path = os.path.join(ARTIFACTS_DIR, "image_index.faiss")
    image_ids = load_json(os.path.join(ARTIFACTS_DIR, "image_ids.json"))
    image_paths = load_json(os.path.join(ARTIFACTS_DIR, "image_paths.json"))
    id_to_path = dict(zip(image_ids, image_paths))
    config = load_json(os.path.join(ARTIFACTS_DIR, "model_config.json"))

    articles = pd.read_csv(CSV_PATH, dtype=str)
    if len(config["text_cols"]) > 1:
        articles["query_text"] = (
            articles[config["text_cols"][0]].fillna("") + " " +
            articles[config["text_cols"][1]].fillna("")
        ).str.strip()
    else:
        articles["query_text"] = articles[config["text_cols"][0]].fillna("")

    article_ids = articles[config["id_col"]].tolist()
    query_texts = articles["query_text"].tolist()

    print(f"[info] Loading OpenCLIP {config['model_name']} ({config['pretrained']})")
    model, _, _ = open_clip.create_model_and_transforms(config["model_name"], pretrained=config["pretrained"])
    model.to(DEVICE).eval()

    text_embeddings = encode_texts_openclip(model, query_texts, DEVICE)

    index = faiss.read_index(faiss_index_path)
    if index.d != text_embeddings.shape[1]:
        raise ValueError("Dim mismatch between index and text embeddings")

    distances, indices = index.search(text_embeddings, 1)
    best_image_ids = [image_ids[int(i[0])] for i in indices]

    mapping_csv = os.path.join(OUT_DIR, "article_best_match.csv")
    safe_makedirs(OUT_DIR)
    with open(mapping_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["article_id", "best_image_id"])
        writer.writerows(zip(article_ids, best_image_ids))

    large_dir = os.path.join(OUT_DIR, LARGE_DIRNAME)
    small_dir = os.path.join(OUT_DIR, SMALL_DIRNAME)
    for d in [large_dir, small_dir]:
        if os.path.exists(d): shutil.rmtree(d)
        safe_makedirs(d)

    for aid, iid in zip(article_ids, best_image_ids):
        src = id_to_path.get(iid, None)
        if src: resize_and_save_png(src, os.path.join(large_dir, f"{aid}_{GROUP_NAME}_{APPROACH_NAME}.png"))

    subset_ids = pd.read_csv(SUBSET_CSV, header=None)[0].astype(str).tolist()
    for aid in subset_ids:
        if aid in article_ids:
            iid = best_image_ids[article_ids.index(aid)]
            src = id_to_path.get(iid, None)
            if src: resize_and_save_png(src, os.path.join(small_dir, f"{aid}_{GROUP_NAME}_{APPROACH_NAME}.png"))

    zip_path = os.path.join(OUT_DIR, f"{GROUP_NAME}.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(OUT_DIR):
            for fn in files:
                if fn.endswith(".png"):
                    full = os.path.join(root, fn)
                    arcname = os.path.relpath(full, OUT_DIR)
                    zf.write(full, os.path.join(GROUP_NAME, arcname))
    print(f"[done] Mapping CSV={mapping_csv}, ZIP={zip_path}")

if __name__ == "__main__":
    main()
