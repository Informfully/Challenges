# NewsImagesRetrieval-MediaEval2025

This project provides three text-to-image retrieval pipelines: 

1.**BLIP Ensemble** -- combines CLIP and BLIP-generated captions 

2.**OpenCLIP Selective** -- retrieval using a selective image set 

3.**OpenCLIP Exhaustive** -- retrieval using all available images

------------------------------------------------------------------------

## 1. File Structure

Recommended repository layout:

    Headline_Hunters/
    │
    ├─ BLIP/
    │   ├─ newsimages_blip_ensemble.py   # Core functions: CLIP/BLIP embeddings, captions, retrieval
    │   ├─ BLIPrun.py                    # Inference: uses weights.json to produce final zip
    │   ├─ BLIPrun.sh                    # Shell wrapper for BLIPrun
    │   ├─ BLIPtrain.py                  # Grid-search to find best weights
    │   ├─ BLIPtrain.sh                  # Shell wrapper for BLIPtrain
    │   ├─ SubsetProcessing.py           # Optional script to create validation subset CSV
    │   └─ requirements.txt              # Python dependencies
    │
    ├─ Selective/
    │   ├─ OpenClipSelective.py          # OpenCLIP selective retrieval + FAISS index
    │   ├─ OCSrun.sh
    │   └─ OCSResultGen.py
    │
    ├─ Exhaustive/
    │   ├─ OpenClipExhaustive.py         # OpenCLIP exhaustive retrieval + FAISS index
    │   ├─ OCErun.sh
    │   └─ OCEResultGen.py
    │
    └─ newsimages_25_v1.1/
        ├─ newsarticles.csv             # Article metadata with article_id, article_title, image_id
        ├─ subset1.csv                  # Small validation set for BLIP training
        └─ newsimages/                  # All images named as <image_id>.jpg/png

------------------------------------------------------------------------

## 2. Dataset Preparation

1.  Download / collect `newsimages_25_v1.1` containing:
    -   `newsarticles.csv` with at least `article_id`, `article_title`,
        and `image_id`.
    -   `subset1.csv` for BLIP weight tuning.
    -   `newsimages/` folder with images named by `image_id`.
2.  Place the dataset in `Headline_Hunters/newsimages_25_v1.1/`.

------------------------------------------------------------------------

## 3. Model Details and Execution

### 3.1 BLIP Ensemble

-   **Workflow**

    -   Understand the text: The program reads each article’s title and tries to fetch its opening paragraph (lead) from the web if available.

    -  Understand the images: It looks at every candidate image and uses an AI model called BLIP to write a short caption describing what it “sees” in each picture.

    -  Compare: All titles, leads, and BLIP captions are turned into numeric “fingerprints” (embeddings) by another model called CLIP.

    -   Mix & Match: A training step figures out how much to trust each clue (title, lead, caption) and combines them to find the best-matching image for each article.

-   **Execution Steps**

    ``` bash
    cd {path to Headline_Hunters}/BLIP/
    pip install -r requirements.txt
    ./BLIPtrain.sh
    ./BLIPrun.sh Headline_Hunters BLIP
    ```

    -   `BLIPtrain.sh` runs `BLIPtrain.py` to produce `weights.json`.
    -   `BLIPrun.sh` runs `BLIPrun.py` to create a submission zip

### 3.2 OpenCLIP Selective

-   **Workflow**

    -   Prepare text: Combines key article fields (like title and tags) into one descriptive string.

    -   Prepare images: Reads all image files mentioned in the CSV (plus any extra files found in the folder).

    -   Create fingerprints: Uses OpenCLIP, a strong image–text model, to turn each article text and each image into comparable fingerprints.

    -   Fast matching: Stores image fingerprints in a FAISS index, a database designed for lightning-fast similarity searches, and measures how well each article text matches each image.

-   **Execution Steps**

    ``` bash
    cd {path to Headline_Hunters}/Selective/
    ./OCSrun.sh
    # For macOS X segmentation faults or Threading issues:
        export OMP_NUM_THREADS=1
        export KMP_DUPLICATE_LIB_OK=TRUE
    # Execute these two before executing ./OCSrun.sh
    python3 OCSResultGen.py
    ```

### 3.3 OpenCLIP Exhaustive

-   **Workflow**

    -   Reads all images in the directory, no matter what the CSV says.

    -   Processes article text and images into fingerprints using OpenCLIP.

    -   Uses FAISS to search every possible image to find the best match for each article.

-   **Execution Steps**

    ``` bash
    cd {path to Headline_Hunters}/Exhaustive/
    ./OCErun.sh
    # For macOS X segmentation faults or Threading issues:
        export OMP_NUM_THREADS=1
        export KMP_DUPLICATE_LIB_OK=TRUE
    # Execute these two before executing ./OCErun.sh
    python3 OCEResultGen.py
    ```

------------------------------------------------------------------------

## 4. Additional Notes

-   **Environment**:
    -   Recommended: Python 3.10+ with a GPU (CUDA or Apple MPS).
    -   Install extras if needed:
        `pip install torch torchvision transformers faiss-cpu open_clip_torch pandas pillow tqdm beautifulsoup4 lxml`.
-   **Artifacts & Outputs**:
    -   BLIP caches intermediate embeddings in `BLIP/cache/`.
    -   OpenCLIP scripts save `image_index.faiss`,
        `image_embeddings.npy`, `model_config.json`, and
        `eval_metrics.json` inside their specified `out_dir`.
-   **File Naming**:
    -   Ensure image files exactly match their `image_id` in the CSV
        (case-sensitive, correct extension).
-   **MacOS**:
    -   Use the provided `export` commands before running `OCSrun.sh` or
        `OCErun.sh` to avoid thread-related segmentation faults.

