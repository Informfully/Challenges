# NewsImages 2025 - Submission Overview

The file 'news_articles_25.xlsx' contains a list of all articles featured in the challenge.
The files 'subtask_1.csv' and 'subtask_2.csv' (see the corresponding subfolders 'subtask_1' and 'subtask_2') list the relevant article IDs that were part of the online survey (a total of 50 articles from among the 8,450 overall articles).

Note that only the image submissions from the online user study are shared in this repository (30 images for subtask 1 and 20 for subtask 2).
The complete archive of all images (>200K) can be downloaded separately from the [file server](https://seafile.ifi.uzh.ch/d/1f14d6d4306340e082d6) and structured as follows:

* **SMALL_GROUPED_BY_ARTICLE**: 30 articles, with 39 images each (group submissions for the small subtask 1, all featured in the online evaluation).
* **LARGE_GROUPED_BY_ARTICLE**: 20 articles, with 26 images each (group submissions for the large subtask 2, all featured in the online evaluation).
* **SMALL_GROUPED_BY_TEAM**: 39 group submissions, each covering the 30 images from the small subtask 1.
* **LARGE_GROUPED_BY_TEAM**: 26 group submissions, each covering up to 8,500 images from the large subtask 2 (contains everything from LARGE_GROUPED_BY_ARTICLE as well as the remaining 8,450 images not featured in the evaluation).

Some teams submitted only one set of images for both subtasks combined.
These submissions were split afterward (separating the 50 articles from subtask 1 from the 8,450 items for subtask 2).
The folders for subtask 1 are labeled 'LARGE_SUBSET' instead of 'SMALL' if they were created this way (i.e., they are a subset of the 'LARGE' submission rather than a separate 'SMALL' submission).

Below is an overview of all participating teams and their submitted runs for Subtask 1 and Subtask 2 (please find the image submissions in the 'subtask_1' and 'subtask_2' folders, respectively).
The images are accompanied by ratings from human evaluators (the leaderboard is shown below; the ratings are available as CSV files in 'survey_results').

## Participating Teams

* AFourP
* CERTH-ITI
* CodingSoft
* CVG-IBA
* DACS-UM-RTL
* Das-RU
* ELITE_CODERS
* Headline Hunters
* SELab-HCMUS
* SSN-CSE
* SyntaxError404
* VisionX

<!--

## Challenge Submissions

Overview of submitted runs for subtask 1 and 2.
Each entry consists of:

1. the name of the team that submitted the run,
2. 'RET' for retrieval approaches and 'GEN' for generative ones, and
3. the name of the approach.

Furthermore, each entry has a unique ID assigned to it for easier identification.
IDs were assigned based on when the paper was submitted (e.g., run 1-01 was from the first submitted paper for subtask 1, run 1-40 from the last one).
The entries of the following lists are sorted alphabetically.

### Subtask 1

* AFourP_RET_OPENCLIP_CLIPRERANK (1-31)
* CERTH_ITI_RET_RUN_1 (1-01)
* CERTH_ITI_RET_RUN_2 (1-02)
* CERTH_ITI_RET_RUN_3 (1-03)
* CERTH_ITI_RET_RUN_4 (1-04)
* CERTH_ITI_RET_RUN_5 (1-05)
* CERTH_ITI_RET_RUN_6 (1-06)
* CodingSoft_GEN_SDTURBO (1-07)
* CodingSoft_RET_CLIP (1-08)
* CVG_IBA_GEN_PromptForge (1-09)
* CVG_IBA_GEN_VIVID (1-10)
* CVG_IBA_RET_SEEK (1-11)
* CVG_IBA_RET_TRACE (1-12)
* DACS_UM_RTL_RET_clip_vit_aesthetics (1-25)
* DACS_UM_RTL_RET_clip_vit_llm_rewriting (1-26)
* DACS_UM_RTL_RET_clip_vit_vlm_judge (1-27)
* Das_RU_GEN_SDXL (1-13)
* Das_RU_GEN_SDXLNEG (1-14)
* Das_RU_GEN_SDXLREF (1-15)
* Das_RU_RET_longtext (1-16)
* Das_RU_RET_reg (1-17)
* Das_RU_RET_sec (1-18)
* ELITE_CODERS_GEN_SDXL (1-19)
* ELITE_CODERS_GEN_STABLE_DIFFUSION (1-20)
* ELITE_CODERS_RET_CLIP (1-21)
* Headline_Hunters_RET_BLIP (1-22)
* Headline_Hunters_RET_OpenClipExhaustive (1-23)
* Headline_Hunters_RET_OpenClipSelective (1-24)
* SELab_HCMUS_GEN_GASumCartoon (1-32)
* SELab_HCMUS_GEN_GASumRealistic (1-33)
* SELab_HCMUS_GEN_HumanSelect (1-34)
* SELab_HCMUS_GEN_LLMCritic (1-35)
* SELab_HCMUS_GEN_LLMCriticTitle (1-36)
* SELab_HCMUS_GEN_VanilaTitleCartoon (1-37)
* SELab_HCMUS_GEN_VanilaTitleRealistic (1-38)
* SSN_CSE_RET_CLIP_KEYBERT_YFCC (1-28)
* SyntaxError404_RET_Retrieval (1-29)
* VisionX_RET_CLIP (1-39)
* VisionX_GEN_SD (1-40)
* Organizers_RET_Baseline (1-30)

### Subtask 2

* CodingSoft_GEN_SDTURBO (2-02)
* CodingSoft_RET_CLIP (2-03)
* CVG_IBA_GEN_PromptForge (2-04)
* CVG_IBA_RET_TRACE (2-05)
* Das_RU_GEN_SDXL (2-06)
* Das_RU_GEN_SDXLNEG (2-07)
* Das_RU_GEN_SDXLREF (2-08)
* Das_RU_RET_longtext (2-25)
* Das_RU_RET_reg (2-26)
* Das_RU_RET_sec (2-27)
* ELITE_CODERS_GEN_SDXL (2-09)
* ELITE_CODERS_RET_CLIP (2-10)
* Headline_Hunters_RET_BLIP (2-17)
* Headline_Hunters_RET_OpenClipExhaustive (2-18)
* Headline_Hunters_RET_OpenClipSelective (2-19)
* SELab_HCMUS_GEN_GASumCartoon (2-11)
* SELab_HCMUS_GEN_GASumRealistic (2-12)
* SELab_HCMUS_GEN_LLMCritic (2-13)
* SELab_HCMUS_GEN_LLMCriticTitle (2-14)
* SELab_HCMUS_GEN_VanilaTitleCartoon (2-15)
* SELab_HCMUS_GEN_VanilaTitleRealistic (2-16)
* SSN_CSE_RET_CLIP_KEYBERT_YFCC (2-20)
* SyntaxError404_RET_Retrieval (2-21)
* VisionX_RET_CLIP (2-23)
* VisionX_GEN_SD (2-24)
* Organizers_RET_BASELINE (2-22)

-->

## Overall Results

The scores below are the average ratings for each image in a run submission (on a 5-point Likert scale).
The "evaluation" folder contains scores for every single image.
The codebase for each team is available in the "workflows" repository.
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
| 1-30 | BASELINE | Original Images | N/A | SMALL | 3.041 |
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
| 1-29 | SyntaxError404 | SyntaxError404_Retrieval (*) | RET | SMALL | 2.593 |
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
| 2-22 | BASELINE | Original Images | N/A | LARGE | 2.956 |
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
| 2-21 | SyntaxError404 | SyntaxError404_Retrieval (*) | RET | LARGE | 2.554 |
| 2-09 | ELITE_CODERS | ELITE_CODERS_CLIP | RET | LARGE | 2.443 |
| 2-02 | CodingSoft | CodingSoft_CLIP | RET | LARGE | 1.933 |
| 2-23 | VisionX | VisionX_SD | GEN | LARGE | 1.717 |

(*) Incomplete submission removed from the final leaderboard.
