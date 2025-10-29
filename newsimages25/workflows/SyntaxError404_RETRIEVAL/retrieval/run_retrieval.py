import os
import numpy as np
from PIL import Image
import torch
import gc
from tqdm import tqdm

# ---------------- CONFIG ----------------
DATA_DIR = 'data'
IMAGES_DIR = os.path.join(DATA_DIR, 'newsimages')
FEATURES_DIR = 'features'
SUBMISSION_DIR = 'submission'
os.makedirs(SUBMISSION_DIR, exist_ok=True)

GROUP_NAME = 'SyntaxError404'
SMALL_SIZE = 30  # small subtask: first 30 articles
IMAGE_OUTPUT_SIZE = (460, 260)  # submission requirement

SMALL_APPROACH = 'RET_Retrieval_SMALL'
LARGE_APPROACH = 'RET_Retrieval_LARGE'

BATCH_SIZE = 128  # batch for similarity computation

# ---------------- LOAD FEATURES ----------------
text_embeds = np.load(os.path.join(FEATURES_DIR, 'text_embeds.npy'))
text_ids = np.load(os.path.join(FEATURES_DIR, 'text_ids.npy'))
image_embeds = np.load(os.path.join(FEATURES_DIR, 'image_embeds.npy'))
image_ids = np.load(os.path.join(FEATURES_DIR, 'image_ids.npy'))

# Convert IDs to strings to avoid KeyError
text_ids = np.array([str(x) for x in text_ids])
image_ids = np.array([str(x) for x in image_ids])

# ---------------- HELPER FUNCTIONS ----------------
def compute_similarity(text_batch, image_embeds):
    """
    Compute cosine similarity between a batch of text embeddings and all image embeddings.
    Returns indices of best matching images for the batch.
    """
    # normalize embeddings
    text_norm = text_batch / np.linalg.norm(text_batch, axis=1, keepdims=True)
    image_norm = image_embeds / np.linalg.norm(image_embeds, axis=1, keepdims=True)
    sim_matrix = np.dot(text_norm, image_norm.T)
    best_idx = np.argmax(sim_matrix, axis=1)
    return best_idx

def save_submission(text_ids_batch, best_image_ids, approach_name):
    out_dir = os.path.join(SUBMISSION_DIR, approach_name)
    os.makedirs(out_dir, exist_ok=True)
    for art_id, img_id in zip(text_ids_batch, best_image_ids):
        img_path = os.path.join(IMAGES_DIR, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        img = img.resize(IMAGE_OUTPUT_SIZE)
        out_path = os.path.join(out_dir, f"{art_id}_{GROUP_NAME}_{approach_name}.png")
        img.save(out_path)
    torch.cuda.empty_cache()
    gc.collect()

# ---------------- SMALL SUBTASK ----------------
print("Generating SMALL subtask results...")
small_text_ids = text_ids[:SMALL_SIZE]
small_text_embeds = text_embeds[:SMALL_SIZE]

best_indices = compute_similarity(small_text_embeds, image_embeds)
best_image_ids = image_ids[best_indices]

save_submission(small_text_ids, best_image_ids, SMALL_APPROACH)

# ---------------- LARGE SUBTASK ----------------
print("Generating LARGE subtask results...")
large_results = {}
for i in tqdm(range(0, len(text_embeds), BATCH_SIZE), desc='LARGE batches'):
    batch_text_embeds = text_embeds[i:i+BATCH_SIZE]
    batch_text_ids = text_ids[i:i+BATCH_SIZE]
    best_idx = compute_similarity(batch_text_embeds, image_embeds)
    best_image_ids = image_ids[best_idx]
    save_submission(batch_text_ids, best_image_ids, LARGE_APPROACH)

print("All submission PNGs generated successfully!")
