import os
import numpy as np
import pandas as pd
import torch
import gc
from encoders import CLIPEncoder  # your existing CLIP encoder

# --- Paths ---
DATA_DIR = 'data'
CSV_FILE = os.path.join(DATA_DIR, 'newsarticles.csv')
FEATURES_DIR = 'features'
os.makedirs(FEATURES_DIR, exist_ok=True)

# --- Device & encoder ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = CLIPEncoder(device=device)

# --- Load dataset ---
df = pd.read_csv(CSV_FILE)
text_list = (df['article_title'] + " " + df['article_tags'].fillna("")).tolist()
article_ids = df['article_id'].to_numpy()

# --- Generate text embeddings in batches ---
BATCH_SIZE = 64  # increase if GPU memory allows
all_text_embeds = []

print("Generating text embeddings in batches...")

for i in range(0, len(text_list), BATCH_SIZE):
    batch = text_list[i:i+BATCH_SIZE]
    emb = encoder.encode_text(batch)
    all_text_embeds.append(emb)
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Processed texts {i} - {i+len(batch)}")

# --- Save embeddings ---
text_embeds = np.vstack(all_text_embeds)
np.save(os.path.join(FEATURES_DIR, 'text_embeds.npy'), text_embeds)
np.save(os.path.join(FEATURES_DIR, 'text_ids.npy'), article_ids)
print(f"Saved {len(text_embeds)} text embeddings to {FEATURES_DIR}")
