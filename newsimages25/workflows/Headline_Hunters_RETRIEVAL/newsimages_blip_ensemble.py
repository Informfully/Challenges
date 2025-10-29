
import os, json, math, time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from PIL import Image
import numpy as np
import os, json, math, time, zipfile

try:
    from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
except Exception as e:
    raise RuntimeError("Please install 'transformers' and required model deps. Error: " + str(e))

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

def fetch_lead(article_url: str, timeout: int = 8) -> Optional[str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; NewsImagesBot/1.0; +https://example.org)"
    }
    try:
        r = requests.get(article_url, timeout=timeout, headers=headers)
        r.raise_for_status()
        html = r.text
        soup = BeautifulSoup(html, "lxml")
        og = soup.find("meta", property="og:description")
        if og and og.get("content"):
            return og.get("content").strip()
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            return meta_desc.get("content").strip()
        article_tag = soup.find("article")
        if article_tag:
            p = article_tag.find("p")
            if p and p.get_text(strip=True):
                return p.get_text(strip=True)
        selectors = ["main", "div[id*='content']", "div[class*='content']", "div[id*='main']",
                     "section[class*='article']", "div[class*='article']", "div[class*='post']"]
        for sel in selectors:
            candidate = soup.select_one(sel)
            if candidate:
                p = candidate.find("p")
                if p and p.get_text(strip=True):
                    return p.get_text(strip=True)
        paragraphs = soup.find_all("p")
        for p in paragraphs:
            text = p.get_text(strip=True)
            if text and len(text) > 40:
                return text
        return None
    except Exception:
        return None

def load_articles_csv(path: str):
    import csv
    rows = []
    # utf-8-sig automatically strips the BOM
    with open(path, newline='', encoding='utf-8-sig') as f:
        dr = csv.DictReader(f)
        for r in dr:
            # normalize keys: strip whitespace
            clean = {k.strip(): v.strip() for k, v in r.items()}
            rows.append(clean)
    return rows

def find_local_images(images_dir: str, rows: List[Dict]) -> Tuple[List[str], List[str]]:
    p = Path(images_dir)
    p.exists() or p.mkdir(parents=True, exist_ok=True)
    found_paths = []
    found_ids = []
    for r in rows:
        aid = r.get("image_id") or ""
        if not aid:
            continue
        for ext in ["jpg", "jpeg", "png"]:
            candidate = p / f"{aid}.{ext}"
            if candidate.exists():
                found_paths.append(str(candidate))
                found_ids.append(aid)
                break
    return found_paths, found_ids

def image_to_numpy(img: Image.Image) -> Image.Image:
    return img.convert("RGB")

def compute_image_embeddings(image_paths: List[str], clip_model, clip_processor, device="cpu", batch_size=32):
    import torch
    model = clip_model
    processor = clip_processor
    model.to(device)
    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            imgs = [image_to_numpy(Image.open(p)) for p in batch]
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            out = model.get_image_features(**inputs)
            arr = out.cpu().numpy()
            arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
            embs.append(arr)
    return np.vstack(embs)

def compute_text_embeddings(texts: List[str], clip_model, clip_processor, device="cpu", batch_size=64):
    import torch
    model = clip_model
    processor = clip_processor
    model.to(device)
    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_text = texts[i:i+batch_size]
            inputs = processor(text=batch_text, return_tensors="pt", padding=True, truncation=True).to(device)
            out = model.get_text_features(**inputs)
            arr = out.cpu().numpy()
            arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
            embs.append(arr)
    return np.vstack(embs)

def generate_blip_captions(image_paths: List[str], blip_model, blip_processor, device="cpu", batch_size=16):
    import torch
    model = blip_model
    processor = blip_processor
    model.to(device)
    model.eval()
    captions = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            imgs = [image_to_numpy(Image.open(p)) for p in batch]
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            out_ids = model.generate(**inputs, max_length=30)
            try:
                caps = processor.decode(out_ids, skip_special_tokens=True)
                if isinstance(caps, str):
                    captions.append(caps)
                else:
                    captions.extend(caps)
            except Exception:
                try:
                    caps = blip_model.config.decoder_start_token_id and processor.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
                    captions.extend(caps)
                except Exception:
                    captions.extend([""] * len(batch))
    return captions

def build_faiss_index(embs: np.ndarray):
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs.astype('float32'))
    return index

def retrieve_top1_for_queries(text_embs_query: Dict[str, np.ndarray],
                              image_embs: np.ndarray,
                              blip_caption_embs: Optional[np.ndarray],
                              weights: Dict[str, float],
                              use_faiss: bool = False):
    w_title = weights.get("title", 0.0)
    w_lead = weights.get("lead", 0.0)
    w_blip = weights.get("blip", 0.0)
    total = w_title + w_lead + w_blip + 1e-12
    w_title /= total; w_lead /= total; w_blip /= total

    title_embs = text_embs_query.get("title")
    lead_embs = text_embs_query.get("lead")
    n_queries = title_embs.shape[0]
    n_images = image_embs.shape[0]

    sims_title = title_embs @ image_embs.T
    sims_lead = lead_embs @ image_embs.T
    if blip_caption_embs is not None:
        sims_blip = title_embs @ blip_caption_embs.T
    else:
        sims_blip = np.zeros_like(sims_title)

    combined = w_title * sims_title + w_lead * sims_lead + w_blip * sims_blip
    top_idxs = np.argmax(combined, axis=1)
    return top_idxs, combined

def save_images_and_zip(out_dir: str, group_name: str, approach_name: str, subtask: str,
                      article_ids: List[str], chosen_image_paths: List[str]):
    from PIL import Image
    out_root = Path(out_dir) / group_name / f"RET_{approach_name}_{subtask}"
    
    # --- FINAL FIX ---
    # Create the full destination directory, not just its parent.
    out_root.mkdir(parents=True, exist_ok=True)
    # --- END OF FIX ---

    for aid, img_path in zip(article_ids, chosen_image_paths):
        try:
            img = Image.open(img_path)
        except Exception:
            img = Image.new("RGB", (460, 260), color=(255, 255, 255))
        
        w, h = img.size
        target_w, target_h = 460, 260
        scale = max(target_w / w, target_h / h)
        new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
        
        img = img.convert("RGB").resize((new_w, new_h), Image.LANCZOS)
        
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        img = img.crop((left, top, left + target_w, top + target_h))
        
        out_path = out_root / f"{aid}_{group_name}_{approach_name}.png"
        img.save(out_path, format="PNG")

    zip_name = f"{group_name}.zip"
    zip_path = Path(out_dir) / zip_name
    
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(Path(out_dir) / group_name):
            for f in files:
                full = Path(root) / f
                # Make sure we only add files to the zip, not the zip itself
                if full.suffix == '.zip':
                    continue
                rel = full.relative_to(Path(out_dir))
                zf.write(full, arcname=str(rel))

    return str(zip_path)
