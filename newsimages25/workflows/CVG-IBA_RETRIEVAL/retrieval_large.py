import os
import sys
import traceback
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import pandas as pd
from PIL import Image, ImageOps
import numpy as np

# --- Helpers ---------------------------------------------------------------

def log_warn(msg):
    print(f"[WARN] {msg}", file=sys.stderr)

def log_err(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)

def safe_call(fn, *args, **kwargs):
    """Call fn and swallow any exception, returning None."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        log_err(f"{fn.__name__} failed: {e}")
        return None

# --- Data loading ----------------------------------------------------------

def load_faiss_index(index_path, ids_path):
    index = faiss.read_index(index_path)
    ids = np.load(ids_path).tolist()
    return index, ids

def load_image_paths(images_dir):
    image_paths = {}
    for fname in os.listdir(images_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_id = os.path.splitext(fname)[0]
            image_paths[image_id] = os.path.join(images_dir, fname)
    return image_paths

# --- Embeddings ------------------------------------------------------------

def embed_text(text, processor, model, device='cpu', max_length=77):
    model.to(device)
    inputs = processor(text=[text], return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype('float32')

def embed_image(image, processor, model, device='cpu'):
    model.to(device)
    inputs = processor(images=image, return_tensors='pt').to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype('float32')

# --- Retrieval -------------------------------------------------------------

def retrieve_images(query, processor, model, index, ids, k=1, device='cpu'):
    text_emb = embed_text(query, processor, model, device)
    distances, indices = index.search(text_emb, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(ids):
            results.append((ids[idx], float(dist)))
    if not results:
        raise ValueError("No results returned from index.search()")
    return results[0]  # top-1

# --- Image processing ------------------------------------------------------

def process_and_save_image(article_id, image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with Image.open(image_path) as im:
        # Respect camera EXIF orientation
        im = ImageOps.exif_transpose(im)

        # Convert incompatible modes to something PNG supports
        if im.mode == "CMYK":
            im = im.convert("RGB")
        elif im.mode in ("P", "LAB", "HSV", "YCbCr"):
            im = im.convert("RGB")
        elif im.mode == "LA":  # L with alpha -> keep alpha
            im = im.convert("RGBA")
        # 'RGB' and 'RGBA' are already fine for PNG

        # Resize with a good resampler
        im = im.resize((460, 260), resample=Image.LANCZOS)

        output_image_path = os.path.join(output_dir, f"{article_id}_CVG-IBA_TRACE.png")

        try:
            im.save(output_image_path, format="PNG")
        except OSError as e:
            # As a last resort, fall back to JPEG (no alpha) if PNG still fails
            fallback = os.path.join(output_dir, f"{article_id}_CVG-IBA_TRACE.jpg")
            im = im.convert("RGB")  # JPEG requires RGB
            im.save(fallback, format="JPEG", quality=95)
            print(f"[WARN] PNG save failed ({e}); saved JPEG instead: {fallback}")
            return fallback

    return output_image_path

# --- Driver ---------------------------------------------------------------

def retrieve_images_for_all_articles(
    df, processor, model, index, ids, images_dir, output_dir='RET_TRACE_LARGE', k=1, device='cpu'
):
    total = len(df)
    ok = 0
    skipped = 0

    for i, row in df.iterrows():
        try:
            article_id = str(row['article_id']).strip()
            title = str(row.get('article_title', '')).strip()
            tags = str(row.get('article_tags', '')).strip()
            query = f"{title} {tags}".strip() or title or tags

            if not query:
                log_warn(f"[row {i}] Empty query for article_id={article_id}; skipping.")
                skipped += 1
                continue

            # Retrieve (top-1)
            try:
                image_id, score = retrieve_images(query, processor, model, index, ids, k, device)
            except Exception as e:
                log_err(f"[row {i}] Retrieval failed for article_id={article_id}: {e}")
                skipped += 1
                continue

            # Determine candidate image path (try jpg/jpeg/png)
            candidates = [
                os.path.join(images_dir, f"{image_id}.jpg"),
                os.path.join(images_dir, f"{image_id}.jpeg"),
                os.path.join(images_dir, f"{image_id}.png"),
            ]
            image_path = next((p for p in candidates if os.path.exists(p)), None)
            if image_path is None:
                log_warn(f"[row {i}] Image file not found for id={image_id}; tried {candidates}")
                skipped += 1
                continue

            # Process & save
            try:
                output_image_path = process_and_save_image(article_id, image_path, output_dir)
            except Exception as e:
                log_err(f"[row {i}] Image processing failed for article_id={article_id}: {e}")
                skipped += 1
                continue

            print(f"Article ID: {article_id}")
            print(f"Retrieved Image Path: {output_image_path}")
            print(f"Similarity Score: {score}\n")
            ok += 1

        except Exception as e:
            # Catch-all so one bad row doesn't stop the loop
            log_err(f"[row {i}] Unexpected error for article_id={row.get('article_id', 'N/A')}: {e}")
            # If you want the stack trace for debugging, uncomment:
            # traceback.print_exc()
            skipped += 1
            continue

    print(f"\nDone. Success: {ok} | Skipped (errors): {skipped} | Total: {total}")

# --- Main -----------------------------------------------------------------

if __name__ == '__main__':
    try:
        # Load model & processor (fine-tuned weights)
        checkpoint_path = 'checkpoints/clip_finetuned_epoch3.pt'
        model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        # Paths
        file_path = 'Dataset/newsarticles.csv'
        newsimages_path = 'Dataset/newsimages/'
        index_path = 'index.faiss'
        ids_path = 'ids.npy'

        # Load data/index — swallow errors for robustness
        df = safe_call(pd.read_csv, file_path)
        if df is None:
            raise RuntimeError("Failed to load CSV; aborting.")

        _ = safe_call(load_image_paths, newsimages_path)  # Not strictly needed later, but harmless
        pair = safe_call(load_faiss_index, index_path, ids_path)
        if pair is None:
            raise RuntimeError("Failed to load FAISS index/ids; aborting.")
        index, ids = pair

        # Run retrieval
        retrieve_images_for_all_articles(
            df, processor, model, index, ids, newsimages_path, k=1, device=device
        )

    except Exception as e:
        log_err(f"Fatal error in main: {e}")
        # traceback.print_exc()
