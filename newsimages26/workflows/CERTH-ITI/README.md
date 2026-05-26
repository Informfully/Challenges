#  Actor-Critic-Image-Generation (ACIG): NewsImages-MediaEval2026 CERTH-ITI
Actor-Critic-Image-Generation (ACIG) is the official implementation (Instructions, code and models) of the CERTH-ITI participation in the MediaEval2026 NewsImages task.

## Introduction
ACIG is a test-time Actor-Critic approach to news images generation: an automated pipeline for generating images from article headlines using a training-free actor-critic feedback loop. Given a set of news article titles, the ACIG approach iteratively produces, evaluates, and refines image generation prompts until the resulting visuals meet a quality threshold - without any human intervention.

This repository presents the CERTH-ITI approach to the NewsImages 2026 challenge, which focuses on generating images that accurately reflect the content and tone of a news article headline. A single-shot prompt often produces visuals that are either too generic or misaligned with the article's key concepts. ACIG addresses this by introducing a self-improving loop inspired by actor-critic reinforcement learning (but, in ACIG, no reinforcement learning or other form of training the actor and critic is performed):

- **Actor** - A large vision-language model reads the article title and generates a descriptive prompt tailored to image generation (and used as input to an image generation model).
- **Image Generator** - One or more diffusion models produce candidate images from the actor's prompt.
- **Critic** - A separate vision-language model evaluates each generated image against the original article headline, assigning a relevance score on a 1–5 scale.
- **Refinement** - If the critic's score falls below a configurable threshold, it feeds the full scoring history back to the actor, which rewrites the prompt with awareness of previous failures. The loop repeats until the threshold is met or the maximum number of iterations is reached.
- **Selection** - The image with the highest critic score across all attempts is selected as the final output for each article.

## Models

ACIG relies on three distinct model roles - **actor**, **critic**, and **image generator** - each fulfilled by a dedicated model. The table below lists the specific models used in the current configuration.



| Role                | Model                           | HuggingFace link                                                               |
|---------------------|---------------------------------|--------------------------------------------------------------------------|
| **Actor**           | `Qwen/Qwen3-VL-8B-Instruct`     | [link](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)                 |
| **Critic**          | `Qwen/Qwen3.5-9B`               | [link](https://huggingface.co/Qwen/Qwen3.5-9B)                           |
| **Image Generator** | `Z-Image-Turbo`                 | [link](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)                  |
| **Image Generator** | `Qwen-Image-SDNQ-uint4-svd-r32` | [link](https://huggingface.co/Disty0/Qwen-Image-2512-SDNQ-uint4-svd-r32) |
| **Image Generator** | `Qwen-Image-2512`               | [link](https://huggingface.co/Qwen/Qwen-Image-2512)                      |


The pipeline is model-agnostic by design - the actor, critic, and image generator components can each be replaced with any model of choice without altering the core method.

## Installation
```bash
pip install -r requirements.txt
```

## Execution Instructions
### Configuration
All experiment's parameters are controlled via the config file (`config.py`) that defines dataset selection, generation settings, model choice, and evaluation behavior.
 
#### Parameters

| Parameter              | Type   | Default  | Description                                                                                                                                          |
|------------------------|--------|----------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `RERUN_INITIAL_PROMPT` | `bool` | `False`  | If `True`, re-runs the initial prompt generation even if output files already exist.                                                                 |
| `THRESHOLD`            | `int`  | `5`      | Score threshold used during iterative refinement.                                                                                                    |
| `ITERATIONS`           | `int`  | `6`      | Maximum number of critic–refine iterations per article before moving on, regardless of whether the score threshold has been reached.                 |
| `IMG_GEN_MODELS`       | `list` | `['zt']` | Image generation backend(s) to use. Supported values: `'zt'`, `'sdnq'`, `'2512'`. Pass multiple values to run all of them sequentially.              |
| `CRITIC_FEEDBACK`      | `bool` | `True`   | If `False`, disables both the critic-feedback and the prompt history during refinement - each iteration generates independently without prior context. |
| `SET`                  | `str`  | `'test'` | Dataset split to process. Accepted values: `'train'` or `'test'`. Automatically resolves the correct CSV path.                                       |


### Execution
```bash
python newsimages2026_certh_iti.py
```
## Citation

If you find our method useful in your work, please cite the following publication where this approach was proposed:


**Citation will be updated upon publication.**
