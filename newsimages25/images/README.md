# NewsImages Submission Overview

Please note that only the image submissions from the online user study are shared in this repository (30 images for subtask 1 and 20 for subtask 2).
The complete archive of all images (>200K) can be downloaded separately from the [file server](https://seafile.ifi.uzh.ch/d/1f14d6d4306340e082d6) and structured as follows:

* **newsimages_25_v1.3.zip**: 8,500 article titles with images from GDELT (this is the final challenge dataset shared with participants)
* **SMALL_GROUPED_BY_ARTICLE**: 30 articles, with 39 images each (group submissions for the small subtask 1, all featured in the online evaluation)
* **LARGE_GROUPED_BY_ARTICLE**: 20 articles, with 26 images each (group submissions for the large subtask 2, all featured in the online evaluation)
* **SMALL_GROUPED_BY_TEAM**: 39 group submissions, each covering the 30 images from the small subtask 1 (more or less identical to SMALL_GROUPED_BY_ARTICLE)
* **LARGE_GROUPED_BY_TEAM**: 26 group submissions, each covering up to 8,500 images from the large subtask 2 (contains everything from LARGE_GROUPED_BY_ARTICLE as well as the remaining 8,450 images not featured in the evaluation)

Below is an overview of all participating teams and their submitted runs for Subtask 1 and Subtask 2 (please find the image submissions in the 'subtask_1' and 'subtask_2' folders, respectively).
The images are provided with the ratings of the human evaluators (the leaderboard is shown below; the ratings are available as CSV files in 'evaluation').

## Participating Teams

```python
TEAMS = ["AFourP",              # AFourP
         "CERTH_ITI",           # CERTH-ITI
         "CodingSoft",          # CodingSoft
         "CVG_IBA",             # CVG-IBA
         "DACS_UM_RTL",         # DACS-UM-RTL
         "Das_RU",              # Das-RU
         "ELITE_CODERS",        # ELITE_CODERS
         "Headline_Hunters",    # Headline Hunters
         "SELab_HCMUS",         # SELab-HCMUS
         "SSN_CSE",             # SSN-CSE
         "SyntaxError404",      # SyntaxError404
         "VisionX"]             # VisionX
```

## Challenge Articles

### Article IDs Subtask 1 ("SMALL")

```python
# Challenge Subtask SMALL
# 30 article IDs (pre-determined)
# Ordering of the original survey preserved
IDS_SMALL = [76, 238, 346, 493, 865,
             1004, 1336, 1817, 2307, 2362,
             2384, 2482, 2509, 2563, 2715,
             2844, 4409, 4609, 5291, 5796,
             5800, 5923, 6074, 6172, 6469,
             7304, 7374, 8192, 8399, 8495]
```

### Article IDs Subtask 2 ("LARGE")

```python
# Challenge Subtask LARGE
# 20 article IDs (randomly selected)
# Ordering of the original survey preserved
IDS_LARGE = [534, 2013, 2411, 3310, 4620,
             5816, 6057, 6834, 7021, 8196,
             440, 1559, 1976, 2154, 3121,
             3918, 4377, 7196, 7601, 8420]
```

## Challenge Submissions

### Subtask 1

```python
# Challenge Subtask SMALL
# 39 runs by 12 teams (excl. baseline) with run ID
# Ordering of the original survey preserved
class RUNS_SMALL(Enum):
    AFourP_RET_OPENCLIP_CLIPRERANK = 31
    CERTH_ITI_RET_RUN_1 = 1
    CERTH_ITI_RET_RUN_2 = 2
    CERTH_ITI_RET_RUN_3 = 3
    CERTH_ITI_RET_RUN_4 = 4
    CERTH_ITI_RET_RUN_5 = 5
    CERTH_ITI_RET_RUN_6 = 6
    CodingSoft_GEN_SDTURBO = 7
    CodingSoft_RET_CLIP = 8
    CVG_IBA_GEN_PromptForge = 9
    CVG_IBA_GEN_VIVID = 10
    CVG_IBA_RET_SEEK = 11
    CVG_IBA_RET_TRACE = 12
    DACS_UM_RTL_RET_clip_vit_aesthetics = 25
    DACS_UM_RTL_RET_clip_vit_llm_rewriting = 26
    DACS_UM_RTL_RET_clip_vit_vlm_judge = 27
    Das_RU_GEN_SDXL = 13
    Das_RU_GEN_SDXLNEG = 14
    Das_RU_GEN_SDXLREF = 15
    Das_RU_RET_longtext = 16
    Das_RU_RET_reg = 17
    Das_RU_RET_sec = 18
    ELITE_CODERS_GEN_SDXL = 19
    ELITE_CODERS_GEN_STABLE_DIFFUSION = 20
    ELITE_CODERS_RET_CLIP = 21
    Headline_Hunters_RET_BLIP = 22
    Headline_Hunters_RET_OpenClipExhaustive = 23
    Headline_Hunters_RET_OpenClipSelective = 24
    SELab_HCMUS_GEN_GASumCartoon = 32
    SELab_HCMUS_GEN_GASumRealistic = 33
    SELab_HCMUS_GEN_HumanSelect = 34
    SELab_HCMUS_GEN_LLMCritic = 35
    SELab_HCMUS_GEN_LLMCriticTitle = 36
    SELab_HCMUS_GEN_VanilaTitleCartoon = 37
    SELab_HCMUS_GEN_VanilaTitleRealistic = 38
    SSN_CSE_RET_CLIP_KEYBERT_YFCC = 28
    SyntaxError404_RET_Retrieval = 29
    VisionX_RET_CLIP = 39
    VisionX_GEN_SD = 40
    Organizers_RET_Baseline = 30
```

### Subtask 2

```python
# Challenge Subtask LARGE
# 26 runs by 9 teams (excl. baseline) with run ID
# Ordering of the original survey preserved
class RUNS_LARGE(Enum):
    AFourP_RET_OPENCLIP_CLIPRERANK = 1
    CodingSoft_GEN_SDTURBO = 2
    CodingSoft_RET_CLIP = 3
    CVG_IBA_GEN_PromptForge = 4
    CVG_IBA_RET_TRACE = 5
    Das_RU_GEN_SDXL = 6
    Das_RU_GEN_SDXLNEG = 7
    Das_RU_GEN_SDXLREF = 8
    Das_RU_RET_longtext = 25
    Das_RU_RET_reg = 26
    Das_RU_RET_sec = 27
    ELITE_CODERS_GEN_SDXL = 9
    ELITE_CODERS_RET_CLIP = 10
    Headline_Hunters_RET_BLIP = 17
    Headline_Hunters_RET_OpenClipExhaustive = 18
    Headline_Hunters_RET_OpenClipSelective = 19
    SELab_HCMUS_GEN_GASumCartoon = 11
    SELab_HCMUS_GEN_GASumRealistic = 12
    SELab_HCMUS_GEN_LLMCritic = 13
    SELab_HCMUS_GEN_LLMCriticTitle = 14
    SELab_HCMUS_GEN_VanilaTitleCartoon = 15
    SELab_HCMUS_GEN_VanilaTitleRealistic = 16
    SSN_CSE_RET_CLIP_KEYBERT_YFCC = 20
    SyntaxError404_RET_Retrieval = 21
    VisionX_RET_CLIP = 23
    VisionX_GEN_SD = 24
    Organizers_RET_BASELINE = 22
```

## Overall Results

Run ID refers to the ID shown above.
The codebase for each team is available in the "workflows" repository.
The scores below are the average ratings for each image in a run submission (ratings are on a 5-point Likert scale).
The "evaluation" folder contains scores for every single image.
Entries are sorted by score (starting with the highest at the top).

### Scores Subtask 1

| Run ID | Team | Description | Type | Subtask | Score |
| - | - | - | - | - | - |
| 12 | CVG-IBA | CVG-IBA_VIVID | GEN | SMALL | 3.401 |
| 9 | CVG-IBA | CVG-IBA_PromptForge | GEN | SMALL | 3.233 |
| 34 | SELab-HCMUS | HCMUSSELab_HumanSelect | GEN | SMALL | 3.115 |
| 22 | Headline Hunters | Headline_Hunters_BLIP | RET | SMALL | 3.096 |
| 11 | CVG-IBA | CVG-IBA_TRACE | RET | SMALL | 3.080 |
| 10 | CVG-IBA | CVG-IBA_SEEK | RET | SMALL | 3.074 |
| 30 | BASELINE | BASELINE | N/A | SMALL | 3.041 |
| 38 | SELab-HCMUS | HCMUSSELab_VanilaTitleRealistic | GEN | SMALL | 3.019 |
| 14 | Das-RU | Das-RU_reg | RET | SMALL | 3.014 |
| 31 | AFourP | AFourP_OPENCLIP_CLIPRERANK | RET | SMALL | 3.008 |
| 13 | Das-RU | Das-RU_longtext | RET | SMALL | 3.006 |
| 23 | Headline Hunters | Headline_Hunters_OpenClipExhaustive | RET | SMALL | 2.999 |
| 37 | SELab-HCMUS | HCMUSSELab_VanilaTitleCartoon | GEN | SMALL | 2.985 |
| 36 | SELab-HCMUS | HCMUSSELab_LLMCriticTitle | GEN | SMALL | 2.985 |
| 17 | Das-RU | Das-RU_SDXLREF | GEN | SMALL | 2.984 |
| 15 | Das-RU | Das-RU_SDXL | GEN | SMALL | 2.971 |
| 16 | Das-RU | Das-RU_SDXLNEG | GEN | SMALL | 2.968 |
| 24 | Headline Hunters | Headline_Hunters_OpenClipSelective | RET | SMALL | 2.965 |
| 19 | ELITE_CODERS | ELITE_CODERS_CLIP | RET | SMALL | 2.964 |
| 33 | SELab-HCMUS | HCMUSSELab_GASumRealistic | GEN | SMALL | 2.940 |
| 21 | ELITE_CODERS | ELITE_CODERS_STABLE_DIFFUSION | GEN | SMALL | 2.896 |
| 6 | CERTH_ITI | CERTH-ITI_RUN_6 | RET | SMALL | 2.893 |
| 32 | SELab-HCMUS | HCMUSSELab_GASumCartoon | RET | SMALL | 2.874 |
| 5 | CERTH_ITI | CERTH-ITI_RUN_5 | RET | SMALL | 2.861 |
| 35 | SELab-HCMUS | HCMUSSELab_LLMCritic | GEN | SMALL | 2.858 |
| 1 | CERTH_ITI | CERTH-ITI_RUN_1 | RET | SMALL | 2.857 |
| 27 | DACS-UM-RTL | um-rtl_clip-vit-large-vlm_judge | RET | SMALL | 2.812 |
| 39 | VisionX | VisionX_CLIP | RET | SMALL | 2.741 |
| 26 | DACS-UM-RTL | um-rtl_clip-vit-large-llm_rewriting | RET | SMALL | 2.712 |
| 18 | Das-RU | Das-RU_sec | RET | SMALL | 2.712 |
| 4 | CERTH_ITI | CERTH-ITI_RUN_4 | RET | SMALL | 2.709 |
| 2 | CERTH_ITI | CERTH-ITI_RUN_2 | RET | SMALL | 2.703 |
| 28 | SSN-CSE | SSN-CSE_CLIP_KEYBERT_YFCC | RET | SMALL | 2.677 |
| 8 | CodingSoft | CodingSoft_SDTURBO | GEN | SMALL | 2.666 |
| 25 | DACS-UM-RTL | um-rtl_clip-vit-large-aesthetics | RET | SMALL | 2.650 |
| 3 | CERTH_ITI | CERTH-ITI_RUN_3 | RET | SMALL | 2.645 |
| 20 | ELITE_CODERS | ELITE_CODERS_SDXL | GEN | SMALL | 2.634 |
| 40 | VisionX | VisionX_SD | GEN | SMALL | 2.607 |
| 29 | SyntaxError404 | SyntaxError404_Retrieval | RET | SMALL | 2.593 |
| 7 | CodingSoft | CodingSoft_CLIP | RET | SMALL | 1.898 |

### Scores Subtask 2

| Run ID | Team | Description | Type | Subtask | Score |
| - | - | - | - | - | - |
| 4 | CVG-IBA | CVG-IBA_PromptForge | GEN | LARGE | 3.194 |
| 5 | CVG-IBA | CVG-IBA_TRACE | RET | LARGE | 3.114 |
| 17 | Headline Hunters | Headline_Hunters_BLIP | RET | LARGE | 3.088 |
| 6 | Das-RU | Das-RU_SDXL | GEN | LARGE | 3.031 |
| 8 | Das-RU | Das-RU_SDXLREF | GEN | LARGE | 3.024 |
| 14 | SELab-HCMUS | HCMUSSELab_LLMCriticTitle | GEN | LARGE | 2.983 |
| 22 | BASELINE | BASELINE | N/A | LARGE | 2.956 |
| 25 | Das-RU | Das-RU_longtext | RET | LARGE | 2.953 |
| 7 | Das-RU | Das-RU_SDXLNEG | GEN | LARGE | 2.938 |
| 15 | SELab-HCMUS | HCMUSSELab_VanilaTitleCartoon | GEN | LARGE | 2.927 |
| 13 | SELab-HCMUS | HCMUSSELab_LLMCritic | GEN | LARGE | 2.923 |
| 26 | Das-RU | Das-RU_reg | RET | LARGE | 2.916 |
| 1 | AFourP | AFourP_OPENCLIP_CLIPRERANK | RET | LARGE | 2.902 |
| 27 | Das-RU | Das-RU_sec | RET | LARGE | 2.893 |
| 19 | Headline Hunters | Headline_Hunters_OpenClipSelective | RET | LARGE | 2.890 |
| 12 | SELab-HCMUS | HCMUSSELab_GASumRealistic | GEN | LARGE | 2.861 |
| 18 | Headline Hunters | Headline_Hunters_OpenClipExhaustive | RET | LARGE | 2.860 |
| 16 | SELab-HCMUS | HCMUSSELab_VanilaTitleRealistic | GEN | LARGE | 2.850 |
| 11 | SELab-HCMUS | HCMUSSELab_GASumCartoon | RET | LARGE | 2.823 |
| 10 | ELITE_CODERS | ELITE_CODERS_SDXL | GEN | LARGE | 2.745 |
| 3 | CodingSoft | CodingSoft_SDTURBO | GEN | LARGE | 2.684 |
| 20 | SSN-CSE | SSN-CSE_CLIP_KEYBERT_YFCC | RET | LARGE | 2.673 |
| 24 | VisionX | VisionX_CLIP | RET | LARGE | 2.612 |
| 21 | SyntaxError404 | SyntaxError404_Retrieval | RET | LARGE | 2.554 |
| 9 | ELITE_CODERS | ELITE_CODERS_CLIP | RET | LARGE | 2.443 |
| 2 | CodingSoft | CodingSoft_CLIP | RET | LARGE | 1.933 |
| 23 | VisionX | VisionX_SD | GEN | LARGE | 1.717 |
