import os
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import pandas as pd
from PIL import Image
import numpy as np

# Load the FAISS index and image paths
def load_faiss_index(index_path, ids_path):
    index = faiss.read_index(index_path)
    ids = np.load(ids_path).tolist()  # Convert numpy array to list for easier handling
    return index, ids

def load_image_paths(images_dir):
    image_paths = {}
    for fname in os.listdir(images_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):  # Consider image extensions
            image_id = os.path.splitext(fname)[0]  # Using the filename without extension as the ID
            image_paths[image_id] = os.path.join(images_dir, fname)
    return image_paths

# Define the text embedding extraction using the pretrained model
def embed_text(text, processor, model, device='cpu', max_length=77):
    model.to(device)
    inputs = processor(text=[text], return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
    
    # Check if truncation happened
    if inputs.input_ids.shape[1] > max_length:
        print(f"Warning: Query '{text[:30]}...' truncated to {max_length} tokens.")
    
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)  # Normalize text embeddings
    return emb.cpu().numpy().astype('float32'), inputs['input_ids'], inputs

# Define the image embedding extraction using the pretrained model
def embed_image(image, processor, model, device='cpu'):
    model.to(device)
    inputs = processor(images=image, return_tensors='pt').to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)  # Normalize image embeddings
    return emb.cpu().numpy().astype('float32')

# Retrieve top-1 most similar image based on text query using the pretrained model
def retrieve_images(query, processor, model, index, ids, k=1, device='cpu'):
    text_emb, _, _ = embed_text(query, processor, model, device)
    distances, indices = index.search(text_emb, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append((ids[idx], float(dist)))
    return results[0]  # Return only the most similar image (top-1)

# Function to process and save the retrieved image
def process_and_save_image(article_id, image_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create directory if it doesn't exist

    # Load the image
    image = Image.open(image_path)

    # Resize the image to 460x260 (landscape orientation)
    resized_image = image.resize((460, 260))

    # Define the output path and save as PNG
    output_image_path = os.path.join(output_dir, f"{article_id}_CVG-IBA_SEEK.png")
    resized_image.save(output_image_path, "PNG")

    return output_image_path

# Main function to handle retrieval for all article IDs in the CSV using the pretrained model
def retrieve_images_for_all_articles(df, processor, model, index, ids, images_dir, output_dir='RET_SEEK_SMALL', k=1, device='cpu'):
    # Loop through all rows in the DataFrame
    for _, row in df.iterrows():
        article_id = str(row['article_id']).strip()
        
        # Generate query using article title and tags
        title = row['article_title']
        tags = row.get('article_tags', '')
        query = f"{title} {tags}".strip()

        # Perform retrieval (top-1 most similar image)
        result = retrieve_images(query, processor, model, index, ids, k, device)
        image_id, score = result

        # Load the image corresponding to the retrieved ID
        image_path = os.path.join(images_dir, f"{image_id}.jpg")

        # Process and save the image in the retrieved_images folder
        output_image_path = process_and_save_image(article_id, image_path, output_dir)

        print(f"Article ID: {article_id}")
        print(f"Retrieved Image Path: {output_image_path}")
        print(f"Similarity Score: {score}")
        print("\n")

# Example of calling the script for all articles in the CSV
if __name__ == '__main__':
    # Load the pretrained model & processor from Hugging Face
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')  # Pretrained CLIP model
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')  # Pretrained CLIP processor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Load your articles CSV and FAISS index

    subset_path = 'Dataset/subset.csv'
    newsimages_path = 'Dataset/newsimages/'
    index_path = 'index.faiss'
    ids_path = 'ids.npy'
    
    df = pd.read_csv(subset_path, header=None, names=["article_id", "article_url", "article_title", "article_tags", "image_id", "image_url"])
    image_paths = load_image_paths(newsimages_path)
    index, ids = load_faiss_index(index_path, ids_path)

    # Call the retrieval function for all articles
    retrieve_images_for_all_articles(df, processor, model, index, ids, newsimages_path, k=1, device=device)