import os
import numpy as np
from PIL import Image
import torch
import gc
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from encoders import CLIPEncoder  # your existing CLIP encoder

# --- Paths ---
DATA_DIR = 'data'
IMAGES_DIR = os.path.join(DATA_DIR, 'newsimages')
CSV_FILE = os.path.join(DATA_DIR, 'newsarticles.csv')
FEATURES_DIR = 'features'
os.makedirs(FEATURES_DIR, exist_ok=True)

# --- Device & encoder ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = CLIPEncoder(device=device)

# --- Load dataset ---
df = pd.read_csv(CSV_FILE)

# --- Helper to load & resize image ---
def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))  # smaller size for faster embeddings
    return img

# --- Prepare image paths ---
image_paths = []
image_ids = []
for _, row in df.iterrows():
    img_path = os.path.join(IMAGES_DIR, f"{row['image_id']}.jpg")
    if os.path.exists(img_path):
        image_paths.append(img_path)
        image_ids.append(row['image_id'])

# --- Generate embeddings in batches ---
BATCH_SIZE = 32
all_image_embeds = []

print("Generating image embeddings in batches...")

for i in range(0, len(image_paths), BATCH_SIZE):
    batch_paths = image_paths[i:i+BATCH_SIZE]
    # Parallel image loading
    with ThreadPoolExecutor() as executor:
        batch_images = list(executor.map(load_image, batch_paths))
    # Encode batch
    emb = encoder.encode_image(batch_images)
    all_image_embeds.append(emb)
    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Processed images {i} - {i+len(batch_paths)}")

# --- Save embeddings ---
image_embeds = np.vstack(all_image_embeds)
np.save(os.path.join(FEATURES_DIR, 'image_embeds.npy'), image_embeds)
np.save(os.path.join(FEATURES_DIR, 'image_ids.npy'), np.array(image_ids))
print(f"Saved {len(image_embeds)} image embeddings to {FEATURES_DIR}")
