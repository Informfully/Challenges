MediaEval 2025 – NewsImages Task
Team
ELITE_CODERS

1. Problem Statement

The MediaEval 2025 NewsImages Task focuses on recommending suitable images for news articles.

We explored two complementary approaches:

Image Retrieval (Primary Task) – selecting the most semantically relevant image from the provided dataset.

Image Generation (Experimental Approach) – generating synthetic editorial-style images using diffusion models when retrieval is not used.

Submission requirements:

One image recommendation per article ID

PNG format images with 460×260 resolution

Naming convention:

[article_id]_ssn-coders_clip.png   (retrieval)  
[article_id]_ssn-coders_sdxl.png   (generation)


Folder structure:

RET_CLIP_SMALLS/  
RET_CLIP_LARGE/  
GEN_SDXL_LARGE/  
GEN_SDXL_SMALL

2. Approaches
🔹 A. Image Retrieval (CLIP + FAISS)

Preprocessing

Loaded the NewsImages 2025 dataset (~8,500 entries).

Cleaned missing values and ensured consistency in IDs.

Feature Extraction

Used OpenAI CLIP (ViT-B/32) to encode article text (title + tags) and images.

Generated joint embeddings by averaging text and image features.

Processed in batches of 500 for efficiency.

Indexing & Retrieval

Stored embeddings in a FAISS index for efficient similarity search.

For each article query, retrieved the most relevant image from the dataset.

Post-processing

Resized retrieved images to 460×260 using Pillow.

Saved results with required naming conventions.

🔹 B. Image Generation (Stable Diffusion XL)

Model Setup

Used stabilityai/stable-diffusion-xl-base-1.0 via the diffusers library.

Adapted prompts from article titles and tags:

"Editorial news photo illustrating: {title}. Keywords: {tags}. 
 Realistic photojournalism, no text, no watermark."


Image Generation

Generated images using StableDiffusionXLPipeline.

Parameters:

num_inference_steps = 25

guidance_scale = 7.5

Output resized to 460×260.

Metadata

Saved prompt information in the PNG metadata for reproducibility.

3. Tools & Technologies

Python 3.10+

PyTorch – model execution

CLIP (ViT-B/32) – retrieval embeddings

FAISS – similarity search

Stable Diffusion XL – generative image synthesis

Pandas, NumPy – dataset handling

Pillow (PIL) – image resizing and saving

4. Execution Steps
Retrieval
pip install -r requirements.txt
python extract_features.py
python retrieve.py


Results → RET_CLIP_SMALL/ and RET_CLIP_LARGE/

Generation
pip install -r requirements.txt
python generate_images.py


Results → GEN_SDXL_LARGE/

5. Final Results

Retrieval Approach: Successfully retrieved relevant images for 8,500+ news articles, aligned with task requirements.

Generation Approach: Produced synthetic editorial-style images for articles, demonstrating an alternative method when retrieval is not sufficient.

Both approaches provide 460×260 PNG outputs, meeting submission guidelines.
