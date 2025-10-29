# README.md

## News Article to Image Retrieval using OpenCLIP
The primary goal of this project is to implement an effective system for retrieving the most relevant image from a large collection for a given news article.

The approach uses a powerful vision-language model (OpenCLIP) in a two-stage process: an initial fast retrieval to find candidate images, followed by a more precise re-ranking to select the best match.

---

### Workflow Overview

The process can be broken down into the following key stages:

1.  **Setup and Initialization**:
    * The necessary Python libraries (`transformers`, `open_clip_torch`, `pandas`, etc.) are installed.
    * The environment is prepared by mounting Google Drive to access the dataset and setting the computation device to a GPU (CUDA) if available.

2.  **Data Loading**:
    * A CSV file named `newsarticles.csv` containing article metadata (ID, title, tags) is loaded into a pandas DataFrame.
    * The path to the `newsimages` folder, which contains the entire collection of images, is defined.

3.  **Image Indexing (Embedding Generation)**:
    * The `ViT-bigG-14` model from OpenCLIP, pre-trained on the LAION-2B dataset, is loaded.
    * To enable fast searching, every image in the `newsimages` folder is pre-processed and converted into a numerical vector representation (an embedding) using the CLIP model's image encoder.
    * These embeddings are normalized and stored in a dictionary (`image_embeddings`) in memory, mapping each `image_id` to its corresponding vector. This is a one-time pre-computation step for the entire image dataset.

4.  **Text Prompt Generation**:
    * For each news article in the DataFrame, a descriptive text prompt is created using the `build_prompt` function. This function combines the `article_title` and the top 5 `article_tags` into a single string (e.g., "News headline: [Title]. Related topics: [Tag1, Tag2, ...].").

5.  **Two-Stage Retrieval Process**:
    This is the core of the retrieval logic, executed for every article:

    * **Stage 1: Candidate Retrieval (Coarse Search)**
        * The text prompt for the article is converted into a text embedding using the OpenCLIP text encoder.
        * This text embedding is compared against all pre-computed image embeddings using cosine similarity.
        * The top 5 images with the highest similarity scores are selected as initial candidates. This method is efficient for quickly narrowing down the search space from thousands of images.

    * **Stage 2: Re-ranking (Fine-grained Search)**
        * The top 5 candidate images are re-evaluated to find the best match.
        * The notebook was designed to use a more sophisticated **BLIP ITM (Image-Text Matching)** model for this step. However, the execution shows that the necessary libraries were not available, so the system automatically used a fallback method.
        * **Fallback Method (CLIP Re-ranking)**: The `rerank_with_clip` function was used. In this step, the text embedding is once again compared via cosine similarity, but only against the top 5 image candidates. The image with the single highest score from this smaller pool is selected as the final result.


---

### Key Components & Models

* **Primary Model (Embedding & Retrieval)**: `open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')`
    * This is a large-scale Vision Transformer model used for generating both image and text embeddings.
* **Re-ranking Model (Intended)**: `Salesforce/blip-itm-base-coco`
    * A BLIP model specialized for Image-Text Matching, intended for more accurate scoring of image-text pairs. *This was not used in the final execution due to import errors.*
* **Re-ranking Model (Actual/Fallback)**: The same `ViT-bigG-14` OpenCLIP model was re-used to score the top 5 candidates.
* **Dataset**:
    * `newsarticles.csv`: A table of news articles.
    * `newsimages/`: A directory containing 8500 JPG images.
* **Core Libraries**: `torch`, `transformers`, `pandas`, `Pillow (PIL)`, `open_clip_torch`.

---