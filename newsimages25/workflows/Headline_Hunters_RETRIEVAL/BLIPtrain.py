import argparse
import json
import os
import csv
from pathlib import Path

import numpy as np
from newsimages_blip_ensemble import (
    compute_image_embeddings,
    compute_text_embeddings,
    fetch_lead,
    find_local_images,
    generate_blip_captions,
    load_articles_csv,
    retrieve_top1_for_queries,
)
script_dir = Path(__file__).resolve().parent
def prepare_embeddings(rows, images_dir, device="cpu"):
    img_paths, img_ids = find_local_images(images_dir, rows)
    if not img_paths:
        raise SystemExit(f"No candidate images found in the specified directory: {images_dir}")
    from transformers import CLIPProcessor, CLIPModel
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image_embs = compute_image_embeddings(img_paths, clip_model, clip_processor, device=device)
    return img_paths, img_ids, image_embs, clip_model, clip_processor

def build_query_embeddings(rows_for_eval, clip_model, clip_processor, device="cpu"):
    titles, leads = [], []
    for r in rows_for_eval:
        title = r.get("article_title", "") or ""
        titles.append(title)
        lead = ""
        if url := r.get("article_url", ""):
            if l := fetch_lead(url):
                lead = l
        if not lead:
            lead = title
        leads.append(lead)

    title_embs = compute_text_embeddings(titles, clip_model, clip_processor, device=device)
    lead_embs = compute_text_embeddings(leads, clip_model, clip_processor, device=device)
    return {"title": title_embs, "lead": lead_embs}

def evaluate_weights(q_embs, image_embs, blip_caption_embs, rows_eval, img_ids, weights):
    top_idxs, _ = retrieve_top1_for_queries(q_embs, image_embs, blip_caption_embs, weights)
    correct = 0
    total = len(rows_eval)
    for i, r in enumerate(rows_eval):
        predicted_img_id = img_ids[top_idxs[i]]
        if str(predicted_img_id) == str(r.get("image_id", "")):
            correct += 1
    return correct / total if total > 0 else 0.0
    
def main(args):
    articles_csv_path = script_dir / args.articles_csv
    images_dir_path = script_dir / args.images_dir
    subset_csv_path = script_dir / args.subset_csv
    out_weights_path = script_dir / args.out_weights

    rows = load_articles_csv(articles_csv_path)
    rows_by_id = {r["article_id"]: r for r in rows if "article_id" in r}
    subset_ids = []
    with open(subset_csv_path, newline='', encoding='utf-8-sig') as f:
        dr = csv.DictReader(f)
        for r in dr:
            if aid := r.get('article_id'):
                subset_ids.append(aid)

    if not subset_ids:
        raise SystemExit(f"Error: Could not read any article IDs from {args.subset_csv}. Is it a valid CSV with an 'article_id' column?")
    rows_eval = [rows_by_id[i] for i in subset_ids if i in rows_by_id]
    if not rows_eval:
        raise SystemExit("Error: None of the article IDs in the subset were found in the main articles CSV. Check for data mismatches.")

    img_paths, img_ids, image_embs, clip_model, clip_processor = prepare_embeddings(rows, images_dir_path, device=args.device)
    q_embs = build_query_embeddings(rows_eval, clip_model, clip_processor, device=args.device)
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        print("Generating BLIP captions...")
        blip_captions = generate_blip_captions(img_paths, blip_model, blip_processor, device=args.device)
        print("Computing BLIP embeddings...")
        blip_caption_embs = compute_text_embeddings(blip_captions, clip_model, clip_processor, device=args.device)
    except Exception as e:
        print(f"Could not process BLIP captions, continuing without them. Error: {e}")
        blip_caption_embs = None

    best = {"score": -1.0, "weights": None}
    steps = args.step
    grid = [i / steps for i in range(steps + 1)]
    print("Starting weight grid search...")
    for w1 in grid:
        for w2 in grid:
            if w1 + w2 > 1.0:
                continue
            w3 = 1.0 - (w1 + w2)
            weights = {"title": w1, "lead": w2, "blip": w3}
            score = evaluate_weights(q_embs, image_embs, blip_caption_embs, rows_eval, img_ids, weights)
            if score > best["score"]:
                best = {"score": score, "weights": weights}
                print(f"New best: {best}")

    with open(out_weights_path, "w", encoding="utf-8") as f:
        json.dump({"best": best}, f, indent=2)
    print(f"Training complete. Best weights written to {out_weights_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles_csv", default="../newsimages_25_v1.1/newsarticles.csv")
    parser.add_argument("--images_dir", default="../newsimages_25_v1.1/newsimages")
    parser.add_argument("--subset_csv", default="../newsimages_25_v1.1/subset1.csv")
    parser.add_argument("--out_weights", default="weights.json")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--step", type=int, default=10)
    args = parser.parse_args()
    main(args)
