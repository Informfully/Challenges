# NewsImages 2025 - Submissions Overview

## Dataset Overview

The files 'subtask_1.csv' and 'subtask_2.csv' (see the corresponding subfolders 'subtask_1' and 'subtask_2') list the relevant article IDs that were part of the online survey (a total of 50 articles from among the 8,450 overall articles).

Note that only the image submissions from the online user study are shared in this repository (39x30 images for subtask 1 and 26x20 images for subtask 2).
The complete archive of all images (>200K) can be downloaded separately from the [file server](https://seafile.ifi.uzh.ch/d/1f14d6d4306340e082d6).
It is structured as follows:

* **SMALL_GROUPED_BY_ARTICLE**: 30 articles, with 39 images each (group submissions for the small subtask 1, all featured in the online evaluation).
* **LARGE_GROUPED_BY_ARTICLE**: 20 articles, with 26 images each (group submissions for the large subtask 2, all featured in the online evaluation).
* **SMALL_GROUPED_BY_TEAM**: 39 group submissions, each covering the 30 images from the small subtask 1.
* **LARGE_GROUPED_BY_TEAM**: 26 group submissions, each covering up to 8,500 images from the large subtask 2 (contains everything from LARGE_GROUPED_BY_ARTICLE as well as the remaining 8,450 images not featured in the evaluation).
* Please note that the other files in the folder are from the [NewsImages 2026](https://github.com/Informfully/Challenges/tree/main/newsimages26/images) iteration.

Some teams submitted only one set of images and used it for both subtasks.
These submissions were split in the archive share above (separating the 50 articles from subtask 1 from the 8,450 items for subtask 2).
The folders for subtask 1 are labeled 'LARGE_SUBSET' instead of 'SMALL' if they were created this way (i.e., they are a subset of the 'LARGE' submission rather than a separate 'SMALL' submission).

Below is an overview of all participating teams and their submitted runs for Subtask 1 and Subtask 2 (please find the image submissions in the 'subtask_1' and 'subtask_2' folders, respectively).
The images are accompanied by ratings from human evaluators (the leaderboard is shown below; the ratings are available as CSV files in 'survey_results').
We also have an [extended image collection](https://github.com/Informfully/Challenges/tree/main/newsimages26/images) available with recommendations of newer workshops.

### News Articles

The file [news_articles_25](https://github.com/Informfully/Challenges/blob/main/newsimages25/images/news_articles_25.xlsx) contains a list of all articles featured in the challenge.

### User Ratings

The folder [survey_results](https://github.com/Informfully/Challenges/tree/main/newsimages25/images/survey_results) contains all user ratings, grouped team and a complete overview file.

## Overall Results

The scores below are the average ratings for each image in a run submission (on a 5-point Likert scale).
The [survey_results](https://github.com/Informfully/Challenges/tree/main/newsimages25/images/survey_results) folder contains scores for every single image.
The codebase for each team is available in the [workflows](https://github.com/Informfully/Challenges/tree/main/newsimages25/workflows) folder.
Entries are sorted by score (starting with the highest at the top).

### Scores Subtask 1

| Run ID | Team | Description | Type | Subtask | Score |
| - | - | - | - | - | - |
| 1-12 | CVG-IBA | CVG-IBA_VIVID | GEN | SMALL | 3.401 |
| 1-09 | CVG-IBA | CVG-IBA_PromptForge | GEN | SMALL | 3.233 |
| 1-34 | SELab-HCMUS | HCMUSSELab_HumanSelect | GEN | SMALL | 3.115 |
| 1-22 | Headline Hunters | Headline_Hunters_BLIP | RET | SMALL | 3.096 |
| 1-11 | CVG-IBA | CVG-IBA_TRACE | RET | SMALL | 3.080 |
| 1-10 | CVG-IBA | CVG-IBA_SEEK | RET | SMALL | 3.074 |
| 1-30 | BASELINE | Original Images | N/A | SMALL | 3.041 (*) |
| 1-38 | SELab-HCMUS | HCMUSSELab_VanilaTitleRealistic | GEN | SMALL | 3.019 |
| 1-14 | Das-RU | Das-RU_reg | RET | SMALL | 3.014 |
| 1-31 | AFourP | AFourP_OPENCLIP_CLIPRERANK | RET | SMALL | 3.008 |
| 1-13 | Das-RU | Das-RU_longtext | RET | SMALL | 3.006 |
| 1-23 | Headline Hunters | Headline_Hunters_OpenClipExhaustive | RET | SMALL | 2.999 |
| 1-37 | SELab-HCMUS | HCMUSSELab_VanilaTitleCartoon | GEN | SMALL | 2.985 |
| 1-36 | SELab-HCMUS | HCMUSSELab_LLMCriticTitle | GEN | SMALL | 2.985 |
| 1-17 | Das-RU | Das-RU_SDXLREF | GEN | SMALL | 2.984 |
| 1-15 | Das-RU | Das-RU_SDXL | GEN | SMALL | 2.971 |
| 1-16 | Das-RU | Das-RU_SDXLNEG | GEN | SMALL | 2.968 |
| 1-24 | Headline Hunters | Headline_Hunters_OpenClipSelective | RET | SMALL | 2.965 |
| 1-19 | ELITE_CODERS | ELITE_CODERS_CLIP | RET | SMALL | 2.964 |
| 1-33 | SELab-HCMUS | HCMUSSELab_GASumRealistic | GEN | SMALL | 2.940 |
| 1-21 | ELITE_CODERS | ELITE_CODERS_STABLE_DIFFUSION | GEN | SMALL | 2.896 |
| 1-06 | CERTH_ITI | CERTH-ITI_RUN_6 | RET | SMALL | 2.893 |
| 1-32 | SELab-HCMUS | HCMUSSELab_GASumCartoon | GEN | SMALL | 2.874 |
| 1-05 | CERTH_ITI | CERTH-ITI_RUN_5 | RET | SMALL | 2.861 |
| 1-35 | SELab-HCMUS | HCMUSSELab_LLMCritic | GEN | SMALL | 2.858 |
| 1-01 | CERTH_ITI | CERTH-ITI_RUN_1 | RET | SMALL | 2.857 |
| 1-27 | DACS-UM-RTL | um-rtl_clip-vit-large-vlm_judge | RET | SMALL | 2.812 |
| 1-39 | VisionX | VisionX_CLIP | RET | SMALL | 2.741 |
| 1-26 | DACS-UM-RTL | um-rtl_clip-vit-large-llm_rewriting | RET | SMALL | 2.712 |
| 1-18 | Das-RU | Das-RU_sec | RET | SMALL | 2.712 |
| 1-04 | CERTH_ITI | CERTH-ITI_RUN_4 | RET | SMALL | 2.709 |
| 1-02 | CERTH_ITI | CERTH-ITI_RUN_2 | RET | SMALL | 2.703 |
| 1-28 | SSN-CSE | SSN-CSE_CLIP_KEYBERT_YFCC | RET | SMALL | 2.677 |
| 1-08 | CodingSoft | CodingSoft_SDTURBO | GEN | SMALL | 2.666 |
| 1-25 | DACS-UM-RTL | um-rtl_clip-vit-large-aesthetics | RET | SMALL | 2.650 |
| 1-03 | CERTH_ITI | CERTH-ITI_RUN_3 | RET | SMALL | 2.645 |
| 1-20 | ELITE_CODERS | ELITE_CODERS_SDXL | GEN | SMALL | 2.634 |
| 1-40 | VisionX | VisionX_SD | GEN | SMALL | 2.607 |
| 1-29 | SyntaxError404 | SyntaxError404_Retrieval | RET | SMALL | 2.593 (**) |
| 1-07 | CodingSoft | CodingSoft_CLIP | RET | SMALL | 1.898 |

### Scores Subtask 2

| Run ID | Team | Description | Type | Subtask | Score |
| - | - | - | - | - | - |
| 2-04 | CVG-IBA | CVG-IBA_PromptForge | GEN | LARGE | 3.194 |
| 2-05 | CVG-IBA | CVG-IBA_TRACE | RET | LARGE | 3.114 |
| 2-17 | Headline Hunters | Headline_Hunters_BLIP | RET | LARGE | 3.088 |
| 2-06 | Das-RU | Das-RU_SDXL | GEN | LARGE | 3.031 |
| 2-08 | Das-RU | Das-RU_SDXLREF | GEN | LARGE | 3.024 |
| 2-14 | SELab-HCMUS | HCMUSSELab_LLMCriticTitle | GEN | LARGE | 2.983 |
| 2-22 | BASELINE | Original Images | N/A | LARGE | 2.956 (*) |
| 2-25 | Das-RU | Das-RU_longtext | RET | LARGE | 2.953 |
| 2-07 | Das-RU | Das-RU_SDXLNEG | GEN | LARGE | 2.938 |
| 2-15 | SELab-HCMUS | HCMUSSELab_VanilaTitleCartoon | GEN | LARGE | 2.927 |
| 2-13 | SELab-HCMUS | HCMUSSELab_LLMCritic | GEN | LARGE | 2.923 |
| 2-26 | Das-RU | Das-RU_reg | RET | LARGE | 2.916 |
| 2-01 | AFourP | AFourP_OPENCLIP_CLIPRERANK | RET | LARGE | 2.902 |
| 2-27 | Das-RU | Das-RU_sec | RET | LARGE | 2.893 |
| 2-19 | Headline Hunters | Headline_Hunters_OpenClipSelective | RET | LARGE | 2.890 |
| 2-12 | SELab-HCMUS | HCMUSSELab_GASumRealistic | GEN | LARGE | 2.861 |
| 2-18 | Headline Hunters | Headline_Hunters_OpenClipExhaustive | RET | LARGE | 2.860 |
| 2-16 | SELab-HCMUS | HCMUSSELab_VanilaTitleRealistic | GEN | LARGE | 2.850 |
| 2-11 | SELab-HCMUS | HCMUSSELab_GASumCartoon | RET | LARGE | 2.823 |
| 2-10 | ELITE_CODERS | ELITE_CODERS_SDXL | GEN | LARGE | 2.745 |
| 2-03 | CodingSoft | CodingSoft_SDTURBO | GEN | LARGE | 2.684 |
| 2-20 | SSN-CSE | SSN-CSE_CLIP_KEYBERT_YFCC | RET | LARGE | 2.673 |
| 2-24 | VisionX | VisionX_CLIP | RET | LARGE | 2.612 |
| 2-21 | SyntaxError404 | SyntaxError404_Retrieval | RET | LARGE | 2.554 (**) |
| 2-09 | ELITE_CODERS | ELITE_CODERS_CLIP | RET | LARGE | 2.443 |
| 2-02 | CodingSoft | CodingSoft_CLIP | RET | LARGE | 1.933 |
| 2-23 | VisionX | VisionX_SD | GEN | LARGE | 1.717 |

(*) Baseline approaches that are not considered for the final ranking.

(**) Incomplete submission removed from the final leaderboard.

If you are using any of these resources, please make sure to properly cite the paper(s) of the team as well as the NewsImages overview paper:

* [NewsImages in MediaEval 2025 – Comparing Image Retrieval and Generation for News Articles](https://github.com/Informfully/Challenges/blob/main/documents/newsimages_task_overview_paper.pdf), Heitz *et al.*, Working Notes Proceedings of the MediaEval 2025 Workshop, 2025.

  ```tex
  @inproceedings{heitz2025newsimages,
  title={NewsImages in MediaEval 2025 – Comparing Image Retrieval and Generation for News Articles},
  author={Heitz, Lucien and Rossetto, Luca and Kille, Benjamin and Lommatzsch, Andreas and Elahi, Mehdi and Dang-Nguyen, Duc-Tien},
  booktitle={Working Notes Proceedings of the MediaEval 2025 Workshop},
  year={2025}
  }
  ```
  