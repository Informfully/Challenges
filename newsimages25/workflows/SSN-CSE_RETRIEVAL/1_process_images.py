import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
from tqdm import tqdm

MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_FOLDER = "yfcc100m_pool"
OUTPUT_FILE = "yfcc_image_embeddings.pt"

print("Loading CLIP model for official run...")
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on device: {device}")

image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"Found {len(image_files)} images to process from the '{IMAGE_FOLDER}' folder.")

image_embeddings = {}

for image_filename in tqdm(image_files, desc="Processing YFCC Images"):
    image_path = os.path.join(IMAGE_FOLDER, image_filename)
    try:
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(device)
            embedding = model.get_image_features(**inputs)
        image_embeddings[image_filename] = embedding.cpu()
    except Exception as e:
        print(f"Could not process image {image_filename}: {e}")

print(f"\nSuccessfully processed {len(image_embeddings)} images.")
print(f"Saving embeddings to {OUTPUT_FILE}...")
torch.save(image_embeddings, OUTPUT_FILE)
print("Done!")