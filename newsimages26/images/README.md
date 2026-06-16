# NewsImages 2026 - Submissions Overview

The file 'news_articles_26.xlsx' contains a list of all articles featured in the challenge.
The subfolder 'survey_articles' contains folders for the relevant article IDs from the online survey (a total of 35 randomly sampled articles from among the 800 overall articles).

Note that only the image submissions from the online user study are shared in this repository (35x30 images).
The complete archive of all images (>20K) can be downloaded separately from the [file server](https://seafile.ifi.uzh.ch/d/1f14d6d4306340e082d6).
It is structured as follows:

* **EXTENSION_GROUPED_BY_ARTICLE**: 35 articles, with 30 images each.
* **EXTENSION_GROUPED_BY_TEAM**: 30 group submissions, covering up to 800 articles each.
* Please note that the other files in the folder are from the [NewsImages 2025](https://github.com/Informfully/Challenges/tree/main/newsimages25/images) iteration.

Below is an overview of all participating teams and their submitted runs.
The images are accompanied by ratings from human evaluators (the leaderboard is shown below; the ratings are available as CSV files in 'survey_results').

## Participating Teams

* AIINS-01
* CERTH-ITI
* FAST-DS
* FAST-MS
* NewsWeavers
* TEAM-A++
* UAmsterdam

## Overall Results

> Note: The images and leaderboard will be published after the workshop, together with the proceedings.

The scores below are the average ratings for each image in a run submission (on a 5-point Likert scale).
The [survey_results](https://github.com/Informfully/Challenges/tree/main/newsimages26/images/survey_results) folder contains scores for every single image.
The codebase for each team is available in the [workflows](https://github.com/Informfully/Challenges/tree/main/newsimages26/workflows) folder.
Entries are sorted by score (starting with the highest at the top).

| Run ID | Team | Description | Type | Score |
| - | - | - | - | - |
| 08 | CERTH-ITI | Run 6 | Generative | 3.616 |
| 04 | CERTH-ITI | Run 2 | Generative | 3.517 |
| 06 | CERTH-ITI | Run 4 | Generative | 3.431 |
| 03 | CERTH-ITI | Run 1 | Generative | 3.429 |
| 07 | CERTH-ITI | Run 5 | Generative | 3.379 |
| 05 | CERTH-ITI | Run 3 | Generative | 3.362 |
| 11 | CERTH-ITI | Run 9 | Generative | 3.355 |
| 02 | CERTH-ITI | Run 10 | Generative | 3.350 |
| 09 | CERTH-ITI | Run 7 | Generative | 3.300 |
| 30 | FAST-MS | RealVISXL | Generative | 3.209 |
| 10 | CERTH-ITI | Run 8 | Generative | 3.197 |
| 33 | Insight | Qwen-Image-Edit-2509 | N/A | 3.085 (**) |
| 31 | Insight | FireRed-Image-Edit-1.1 | N/A | 3.044 (**) |
| 25 | TEAM-A++ | SDXL-CFT | Generative | 3.041 |
| 32 | Insight | Flux.2 Klein | N/A | 2.990 (**) |
| 23 | NewsWeavers | Qwen 3B Plan Prompt Zero Shot FLUX | Generative | 2.907 |
| 29 | BASELINE | Original Images | N/A | 2.880 (*) |
| 21 | NewsWeavers | Qwen 3B Plan LoRA FLUX | Generative | 2.826 |
| 24 | NewsWeavers | Qwen 3B Plan Prompt Zero Shot SDXL | Generative | 2.636 |
| 28 | UAmsterdam | Z-Image Caricature | Generative | 2.563 (*) |
| 17 | FAST-DS | VisionAnchor | Retrieval | 2.390 |
| 22 | NewsWeavers | Qwen 3B Plan LoRA SDXL | Generative | 2.331 |
| 27 | UAmsterdam | Qwen Color Caricature | Hybrid | 2.330 (*) |
| 15 | FAST-DS | QueryForge | Retrieval | 2.254 |
| 16 | FAST-DS | VisionAnchor-R | Retrieval | 2.206 |
| 18 | FAST-MS | Basic Retrieval | Retrieval | 2.205 |
| 13 | FAST-DS | QueryForge-VGQE | Retrieval | 2.187 |
| 12 | FAST-DS | QueryForge-RX | Retrieval | 2.176 |
| 19 | FAST-MS | Hybrid Retrieval | Hybrid | 2.162 |
| 20 | FAST-MS | Official Retrieval | Retrieval | 2.112 |
| 26 | UAmsterdam | Flux Black-White Caricature | Hybrid | 1.972 (*) |
| 14 | FAST-DS | QueryForge-VGRX | Retrieval | 1.831 |
| 01 | AIINS-01 | PD12MTFIDF | Retrieval | 1.614 |

(*) Baseline approaches that are not considered for the final ranking.

(**) Quest for Insight approaches generated for historical news articles that are not considered for the final ranking.
