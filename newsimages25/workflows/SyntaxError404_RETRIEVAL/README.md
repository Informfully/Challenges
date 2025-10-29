# MediaEval 2025 NewsImages Retrieval Project

**Group Name:** SyntaxError404  
**Task:** NewsImages 2025 – Image Retrieval for News Articles  
**Description:** This project implements an image retrieval pipeline that recommends the most fitting news image for a given news article.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Features](#features)  
- [Setup](#setup)  
- [Usage](#usage)  
- [Submission](#submission)  
- [Notes](#notes)  

---

## Project Overview

This project implements a pipeline for the **NewsImages 2025 challenge**. The pipeline consists of:

1. **Text embedding generation** (`text_encoder.py`)  
2. **Image embedding generation** (`image_encoder.py`)  
3. **Retrieval pipeline** (`run_retrieval.py`)  

The retrieval pipeline computes similarities between text and image embeddings and generates **submission-ready PNGs** for both **SMALL (30 articles)** and **LARGE (all articles)** subtasks.

---

## Dataset

- **News articles CSV:** `data/newsarticles.csv`  
- **News images:** `data/newsimages/` folder with `.jpg` files  
- **Embeddings:** Stored in `features/` after running encoders  

> Note: Large datasets, embeddings, and outputs are **not included in this repo**.  

---

## Features

- **Text embeddings:** `features/text_embeds.npy`, `features/text_ids.npy`  
- **Image embeddings:** `features/image_embeds.npy`, `features/image_ids.npy`  
- **Submission images:** Generated in `submission/`  

- PNGs are **460×260 pixels** and named as `[articleID]_[GroupName]_[ApproachName].png`  
- Approach names: `RET_Retrieval_SMALL` and `RET_Retrieval_LARGE`  

---

## Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/mediaeval-newsimages.git
cd mediaeval-newsimages

file structure :

mediaeval_project/
├── data/
│   ├── newsarticles.csv
│   └── newsimages/           # All JPG images
├── features/
│   ├── image_embeds.npy
│   ├── image_ids.npy
│   ├── text_embeds.npy
│   └── text_ids.npy
├── submission/
│   ├── RET_Retrieval_SMALL/
│   └── RET_Retrieval_LARGE/
├── retrieval/
│   ├── preprocess.py
│   ├── image_encoder.py        # generates image embeddings
│   ├── text_encoder.py         # generates text embeddings
│   ├── run_retrieval.py
│   └── utils.py            
├── requirements.txt
└── README.md
