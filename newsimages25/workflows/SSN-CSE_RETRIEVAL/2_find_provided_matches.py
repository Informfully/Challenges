import torch
import torch.nn.functional as F
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import json
import spacy
from keybert import KeyBERT

MODEL_NAME = "openai/clip-vit-base-patch32"
CSV_FILE = "newsimages_25_v1.0/newsarticles.csv"
IMAGE_EMBEDDINGS_FILE = "yfcc_image_embeddings.pt" 
OUTPUT_FILE = "retrieval_results_yfcc.json" 

# Keyword Extraction 
print("Loading spaCy + KeyBERT...")
nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT()

def extract_visual_keywords(text, top_n=5):
    """Extract visual entities and keywords from text."""
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents 
                if ent.label_ in ["ORG", "GPE", "LOC", "EVENT", "FAC", "PRODUCT", "ANIMAL"]]

    kw = kw_model.extract_keywords(text, 
                                   keyphrase_ngram_range=(1, 2), 
                                   stop_words='english', 
                                   top_n=top_n)
    keywords = [k[0] for k in kw]

    combined = list(dict.fromkeys(entities + keywords))
    return combined[:top_n]

# CLIP Model + Embeddings

print("Loading CLIP model + pre-computed image embeddings...")
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image_embeddings_map = torch.load(IMAGE_EMBEDDINGS_FILE)
image_ids = list(image_embeddings_map.keys())
image_embeddings_tensor = torch.cat([image_embeddings_map[img_id] for img_id in image_ids], dim=0).to(device)

# Load Articles
df = pd.read_csv(CSV_FILE)
articles_df = df.drop_duplicates(subset=['article_id'])
print(f"Found {len(articles_df)} unique articles to process.")

results = {}


for _, row in tqdm(articles_df.iterrows(), total=len(articles_df), desc="Matching Articles to Images"):
    article_id = row['article_id']
    title = str(row['article_title'])
    tags = str(row['article_tags'])

    text_context = f"{title}. {tags}"
    keywords = extract_visual_keywords(text_context, top_n=5)

    if keywords:
        text_prompt = f"{title}. Keywords: {', '.join(keywords)}"
    else:
        text_prompt = title

    with torch.no_grad():
        inputs = processor(text=text_prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        text_embedding = model.get_text_features(**inputs)

    similarities = F.cosine_similarity(text_embedding, image_embeddings_tensor)
    best_match_index = torch.argmax(similarities)
    best_image_id = image_ids[best_match_index]

    results[article_id] = best_image_id



print(f"\nFinished matching. Saving trial results to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=4)
print("✅ Done!")
