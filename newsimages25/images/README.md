# NewsImages Submission Overview

*Overview generated on the basis of the evaluaiton script.*

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
# 39 runs by 12 teams (exlc. baseline)
# Ordering of the original survey preserved
class RUNS_SMALL(Enum):
    AFourP_RET_OPENCLIP_CLIPRERANK = 31              # Updated submission
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
    DACS_UM_RTL_RET_clip_vit_aesthetics = 25        # Updated submission
    DACS_UM_RTL_RET_clip_vit_llm_rewriting = 26     # Updated submission
    DACS_UM_RTL_RET_clip_vit_vlm_judge = 27         # Updated submission
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
    SELab_HCMUS_GEN_GASumCartoon = 32               # Updated submission
    SELab_HCMUS_GEN_GASumRealistic = 33             # Updated submission
    SELab_HCMUS_GEN_HumanSelect = 34                # Updated submission
    SELab_HCMUS_GEN_LLMCritic = 35                  # Updated submission
    SELab_HCMUS_GEN_LLMCriticTitle = 36             # Updated submission
    SELab_HCMUS_GEN_VanilaTitleCartoon = 37         # Updated submission
    SELab_HCMUS_GEN_VanilaTitleRealistic = 38       # Updated submission
    SSN_CSE_RET_CLIP_KEYBERT_YFCC = 28
    SyntaxError404_RET_Retrieval = 29
    VisionX_RET_CLIP = 39                           # Updated submission
    VisionX_GEN_SD = 40                             # Updated submission
    Organizers_RET_Baseline = 30                    # Baseline
```

### Subtask 2

```python
# Challenge Subtask LARGE
# 26 runs by 9 teams (exlc. baseline)
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
    Das_RU_RET_longtext = 25                        # Updated submission
    Das_RU_RET_reg = 26                             # Updated submission
    Das_RU_RET_sec = 27                             # Updated submission
    ELITE_CODERS_GEN_SDXL = 9
    ELITE_CODERS_RET_CLIP = 10
    Headline_Hunters_RET_BLIP = 17
    Headline_Hunters_RET_OpenClipExhaustive = 18
    Headline_Hunters_RET_OpenClipSelective = 19
    SELab_HCMUS_GEN_GASumCartoon = 11               # Updated submission
    SELab_HCMUS_GEN_GASumRealistic = 12             # Updated submission
    SELab_HCMUS_GEN_LLMCritic = 13                  # Updated submission
    SELab_HCMUS_GEN_LLMCriticTitle = 14             # Updated submission
    SELab_HCMUS_GEN_VanilaTitleCartoon = 15         # Updated submission
    SELab_HCMUS_GEN_VanilaTitleRealistic = 16       # Updated submission
    SSN_CSE_RET_CLIP_KEYBERT_YFCC = 20
    SyntaxError404_RET_Retrieval = 21
    VisionX_RET_CLIP = 23                           # Updated submission
    VisionX_GEN_SD = 24                             # Updated submission
    Organizers_RET_BASELINE = 22                    # Baseline
```
