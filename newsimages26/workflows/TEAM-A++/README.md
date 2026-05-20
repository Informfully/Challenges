# 2026 MediaEval NewsImages - Image Generation Workflow

This notebook generates a news-thumbnail image for every article in the test set.
For each article it samples 5 candidates with SDXL-Turbo, rejects candidates that
contain rendered text, and keeps the candidate with the highest CFT-CLIP
image-text similarity score (best-of-5). All experiment and analysis code has
been removed; only the generation workflow remains.

Notebook file: `cs5262_newsimages_2026_image_generation.ipynb`

## Prerequisites

- Runtime: Google Colab with a GPU (tested on T4).
- HuggingFace token stored as a Colab secret named `HF_TOKEN`
  (Colab left panel -> key icon -> add `HF_TOKEN`).
- Input file `news_articles_test.csv` uploaded to the Colab working directory,
  with columns `article_id` and `article_title`.

## Models

| Model | Role |
| --- | --- |
| SDXL-Turbo (`stabilityai/sdxl-turbo`) | 1-step image generator |
| EasyOCR | Detects rendered text; candidates with text are rejected |
| OpenAI CLIP ViT-B/32 | Baseline image-text scorer |
| CFT-CLIP (`humane-lab/CFT-CLIP`) | News-specific scorer used to select the best candidate |

## Workflow

```
news_articles_test.csv
        |
        v
[Cell 1] Install packages
        |
        v
[Cell 2] Initialize models  (SDXL-Turbo, EasyOCR, CLIP, CFT-CLIP)
        |
        v
[Cell 3] Define pipeline functions  (clean_title, generate_thumbnail, ...)
        |
        v
[Step 1] For each article:
           clean title
           -> generate 5 candidates (seed = GENERATOR_SEED + i)
           -> reject candidates containing text (OCR)
           -> select best candidate by CFT-CLIP score
           -> save PNG to thumbnails_best/{article_id}.png
        |
        v
[Step 2] Write generation_results.csv
         -> package thumbnails into TEAM_A++_Submission.zip
         -> download the ZIP
```

### Step-by-step

| Cell | Name | What it does |
| --- | --- | --- |
| 1 | Install packages | Installs `diffusers`, `transformers`, `open_clip_torch`, `easyocr`, `ftfy`, and dependencies. |
| 2 | Initialize models | Logs in to HuggingFace Hub and loads SDXL-Turbo, EasyOCR, standard CLIP, and CFT-CLIP onto the GPU. |
| 3 | Core pipeline functions | Defines title cleaning, prompt sanitization, the OCR text filter, CLIP/CFT-CLIP scorers, embedding helpers, and `generate_thumbnail()`. |
| 4 (Step 1) | Generate thumbnails | Reads the test CSV, runs best-of-5 generation per article, and saves each selected thumbnail to `thumbnails_best/`. |
| 5 (Step 2) | Save and package | Writes `generation_results.csv` and packages all thumbnails into a submission ZIP, then downloads it. |

## Generation logic (`generate_thumbnail`)

1. Build the prompt: `news image thumbnail, illustration, vector art, {headline}`.
2. Generate `n` candidates. Candidate `i` uses seed `GENERATOR_SEED + i`, so
   re-running the notebook produces identical images.
3. For each candidate, run EasyOCR. If high-confidence text is detected, the
   candidate is rejected.
4. Score all non-rejected candidates with CFT-CLIP cosine similarity.
5. Return the candidate with the highest CFT-CLIP score.
6. Fallback: if every candidate is rejected, return the first generated image.

## Key parameters

| Parameter | Value | Location |
| --- | --- | --- |
| `GENERATOR_SEED` | 42 | Cell 3 |
| `N_CANDIDATES` | 5 | Step 1 |
| Candidate seed | `GENERATOR_SEED + i` | `generate_thumbnail()` |
| Generation size | 512x512, resized to 460x260 | `generate_thumbnail()` |
| OCR reject threshold | 0.5 confidence | `has_text()` |
| Inference steps | 1, guidance scale 0.0 | `generate_thumbnail()` |

## Outputs

| File | Contents |
| --- | --- |
| `thumbnails_best/{article_id}.png` | One selected thumbnail per article (460x260) |
| `generation_results.csv` | `article_id`, `headline`, `cft_score`, `thumbnail_path` |
| `TEAM_A++_Submission.zip` | All thumbnails plus `generation_results.csv` |

## How to run

1. Open the notebook in Google Colab.
2. Set the runtime to a GPU instance.
3. Add the `HF_TOKEN` Colab secret.
4. Upload `news_articles_test.csv` to the working directory.
5. Run all cells in order. The submission ZIP downloads automatically at the end.

## Reproducibility

Image generation is seeded. With the same input CSV, the same models, and the
same `GENERATOR_SEED`, the notebook produces identical thumbnails on every run.
