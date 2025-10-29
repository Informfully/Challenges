
import argparse
import json
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

from newsimages_blip_ensemble import(
    load_articles_csv, find_local_images, compute_image_embeddings,
    compute_text_embeddings, generate_blip_captions,
    retrieve_top1_for_queries, save_images_and_zip, fetch_lead
)

def main(args):
    CACHE_DIR = Path("cache")
    CACHE_DIR.mkdir(exist_ok=True)
    
    IMAGE_EMBS_CACHE = CACHE_DIR / "image_embs.npy"
    LEADS_CACHE = CACHE_DIR / "leads.json"
    TITLE_EMBS_CACHE = CACHE_DIR / "title_embs.npy"
    LEAD_EMBS_CACHE = CACHE_DIR / "lead_embs.npy"
    BLIP_CAPTIONS_CACHE = CACHE_DIR / "blip_captions.json"
    BLIP_EMBS_CACHE = CACHE_DIR / "blip_embs.npy"

    rows = load_articles_csv(args.articles_csv)
    article_ids = [r['article_id'] for r in rows]
    img_paths, img_ids = find_local_images(args.images_dir, rows)

    if len(img_paths) == 0:
        raise SystemExit("No images found in images_dir")

    from transformers import CLIPProcessor, CLIPModel
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.to(args.device)

    if IMAGE_EMBS_CACHE.exists():
        print("Loading cached image embeddings...")
        image_embs = np.load(IMAGE_EMBS_CACHE)
    else:
        print("Computing image embeddings...")
        image_embs = compute_image_embeddings(img_paths, clip_model, clip_processor, device=args.device)
        np.save(IMAGE_EMBS_CACHE, image_embs)
        print(f"Saved image embeddings to {IMAGE_EMBS_CACHE}")

    titles = [r.get("article_title", "") or "" for r in rows]
    if LEADS_CACHE.exists():
        print("Loading cached article leads...")
        with open(LEADS_CACHE, 'r', encoding='utf-8') as f:
            leads = json.load(f)
    else:
        print("Fetching article leads....") 
        leads = []
        for r in tqdm(rows, desc="Fetching Article Leads"):
            url = r.get("article_url", "") or ""
            lead = ""
            if url:
                lead = fetch_lead(url) or ""
            if not lead:
                lead = r.get("article_title", "") # Fallback to title
            leads.append(lead)
        with open(LEADS_CACHE, 'w', encoding='utf-8') as f:
            json.dump(leads, f)
        print(f"Saved article leads to {LEADS_CACHE}")

    if TITLE_EMBS_CACHE.exists():
        print("Loading cached title embeddings...")
        title_embs = np.load(TITLE_EMBS_CACHE)
    else:
        print("Computing title embeddings...")
        title_embs = compute_text_embeddings(titles, clip_model, clip_processor, device=args.device)
        np.save(TITLE_EMBS_CACHE, title_embs)
        print(f"Saved title embeddings to {TITLE_EMBS_CACHE}")

    if LEAD_EMBS_CACHE.exists():
        print("Loading cached lead embeddings...")
        lead_embs = np.load(LEAD_EMBS_CACHE)
    else:
        print("Computing lead embeddings...")
        lead_embs = compute_text_embeddings(leads, clip_model, clip_processor, device=args.device)
        np.save(LEAD_EMBS_CACHE, lead_embs)
        print(f"Saved lead embeddings to {LEAD_EMBS_CACHE}")
    q_embs = {"title": title_embs, "lead": lead_embs}

    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model.to(args.device)

        if BLIP_CAPTIONS_CACHE.exists():
            print("Loading cached BLIP captions...")
            with open(BLIP_CAPTIONS_CACHE, 'r', encoding='utf-8') as f:
                blip_captions = json.load(f)
        else:
            print("Generating BLIP captions...")
            blip_captions = generate_blip_captions(img_paths, blip_model, blip_processor, device=args.device)
            with open(BLIP_CAPTIONS_CACHE, 'w', encoding='utf-8') as f:
                json.dump(blip_captions, f)
            print(f"Saved BLIP captions to {BLIP_CAPTIONS_CACHE}")

        if BLIP_EMBS_CACHE.exists():
            print("Loading cached BLIP embeddings...")
            blip_caption_embs = np.load(BLIP_EMBS_CACHE)
        else:
            print("Computing BLIP embeddings...")
            blip_caption_embs = compute_text_embeddings(blip_captions, clip_model, clip_processor, device=args.device)
            np.save(BLIP_EMBS_CACHE, blip_caption_embs)
            print(f"Saved BLIP embeddings to {BLIP_EMBS_CACHE}")

    except Exception as e:
        print(f"Could not process BLIP captions. Error: {e}")
        blip_caption_embs = None

    print("Loading weights and running retrieval...")
    with open(args.weights, 'r', encoding='utf-8') as f:
        w = json.load(f)
    weights = w.get("best", {}).get("weights", {"title": 0.3, "lead": 0.6, "blip": 0.1})
    top_idxs, _ = retrieve_top1_for_queries(q_embs, image_embs, blip_caption_embs, weights)
    chosen_image_paths = [img_paths[i] for i in top_idxs if i < len(img_paths)]
    print("Saving images and creating submission ZIP...")
    zip_path = save_images_and_zip(args.out_dir, args.group_name, args.approach_name, args.subtask, article_ids, chosen_image_paths)
    print(f"Submission ZIP created at: {zip_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval with caching.")
    parser.add_argument("--articles_csv", default="../newsimages_25_v1.1/newsarticles.csv")
    parser.add_argument("--images_dir", default="../newsimages_25_v1.1/newsimages")
    parser.add_argument("--weights", default="weights.json")
    parser.add_argument("--group_name", required=True)
    parser.add_argument("--approach_name", required=True)
    parser.add_argument("--subtask", default="LARGE")
    parser.add_argument("--out_dir", default="submissions_out")
    parser.add_argument("--device", default="cpu", help="Device to run on (e.g., 'cpu', 'mps', 'cuda')")
    args = parser.parse_args()
    main(args)
