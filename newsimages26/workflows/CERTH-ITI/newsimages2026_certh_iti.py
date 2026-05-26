import shutil
from dataclasses import dataclass, field
from typing import List, Dict
import pandas as pd
import pickle
import torch
import json
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
import time
from pathlib import Path
import config

from article_models import Article_26, PromptEntry, articles_to_json

from models.Qwen3_VL_8B_Instruct import Qwen3VLAssistant
from models.z_image_generator import ZImageGenerator
from models.Qwen_35_9B import Qwen359BVLAssistant
from models.Qwen_sdnq_uint4 import QwenImageGenerator
from models.Qwen_2512 import QwenImage2512Generator
from models.Flux2_Klein import Flux2KleinGenerator


def is_articles_complete(articles: List[Article_26]):
    c=0
    for i, article in enumerate(articles):
        if article.is_complete:
            c+=1
        # print(f"{i}/{c}")
    if c-1 == i:
        return True
    else:
        return False


def refinement( articles: List[Article_26],
    used_prompt: str,
    attempt: int
    ):
    # ──────────────────────────────────────────
    # LOAD MODEL
    # ──────────────────────────────────────────
    print(f"Loading {MODEL_ID_LLM} … (first run downloads the weights, ~16 GB)")
    # ── Load ──────────────────────────────────────────────────────────────────────
    assistant = Qwen3VLAssistant()
    assistant.load()
    print("Model loaded.\n")

    for i, article in enumerate(articles):
        if article.is_complete:
            continue
        if max(article.prompts[used_prompt][attempt].scores) >= config.THRESHOLD:
            article.is_complete = True
            continue

        if config.CRITIC_FEEDBACK:
            history = []
            for attms in article.prompts[used_prompt]:
                print(attms.text)
                print(attms.scores)
                history.append((attms.text, max(attms.scores)))

            history_str = "\n".join(
                f"  Attempt {j + 1}: Prompt: {p!r} | Score: {s}"
                for j, (p, s) in enumerate(history)
            )
            new_prompt = assistant.ask_for_improved_prompt(history_str=history_str, article_title=article.title)
            ## TODO ADD NEW PROMPTENTRY into article
            articles[i].prompts[used_prompt].append(PromptEntry(text=new_prompt))
        else:
            context = f"""
                {article.title}
                """

            ## prompt#1: Using this article title create a prompt for an image generation model to create a relative to the article image
            question = (prompts[used_prompt])

            # answer = ask_about_text(context, question)
            new_prompt = assistant.ask_about_text(context=context, question=question, max_new_tokens=512, temperature=0.3,
                                              top_p=0.9, )
            articles[i].prompts[used_prompt].append(PromptEntry(text=new_prompt))


def critic(
        # VLModel: str,
        articles: List[Article_26],
        used_prompt: str,
        attempt: int
):
    # ── Load ──────────────────────────────────────────────────────────────────────
    assistant = Qwen359BVLAssistant()
    assistant.load()

    for i, article in enumerate(articles):
        if not article.is_complete:
            # ── Ask about an image ────────────────────────────────────────────────────────
            # title = "Internet Outage Highlights Lingering Broadband Disparity"
            title = article.title
            IMAGE_PATHS = article.prompts[used_prompt][attempt].image_paths

            for image_path in tqdm(IMAGE_PATHS):

                IMAGE_QUESTION = (f"Rate 1–5 how accurately this image captures the key attributes of "
                                  f"the text article without depicting any important elements not present in the article given the headline: '{title}'. "
                                    f"Reply with a single integer between 1 and 5. Do not explain. Do not add any other text. "
                                    f"Your entire response must be exactly one digit."
                                  f"Provide your rate using a single score. ")

                image_answer = assistant.ask_about_image(
                    image=image_path,
                    question=IMAGE_QUESTION,
                    max_new_tokens=50000,
                    temperature=0.3,
                    top_p=0.9,
                    enable_thinking=False,
                )
                article.prompts[used_prompt][attempt].add_score(int(image_answer))
                # print("Image answer:", image_answer)
                print(f"{article.id} / Image score: {image_answer}")


def image_gen_zturbo(
        articles: List[Article_26],
        used_prompt: str,
        attempt: int,
        output_data:  dict = None
):
    generator = ZImageGenerator(device="cuda")
    generator.load()

    negative_prompt = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"

    # negative_prompt = " " # using an empty string if you do not have specific concept to remove
    # output_data = []
    start = time.time()
    for i, article in enumerate(articles):
        if not article.is_complete:

            prompt = article.prompts[used_prompt][attempt].text
            print(f"{article.id} / {prompt}")
            # ── Generate ──────────────────────────────────────────────────────────────────
            image = generator.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                aspect_ratio="16:9",
                num_inference_steps = 20,
                seed=42,
            )
            p = Path(f"{config.RUN_ID}/{MODEL_ID_LLM.split('/')[-1]}/Z-Image-Turbo/{used_prompt}/{attempt}")
            p.mkdir(parents=True, exist_ok=True)
            folder_path = str(p)
            image.save(f"{folder_path}/{article.id}.png")
            article.prompts[used_prompt][attempt].add_image(f"{folder_path}/{article.id}.png")

            if article.id not in output_data:
                # First time seeing this id — create the entry
                output_data[article.id] = {
                    "id": article.id,
                    "title": article.title,
                    "images": [f"{folder_path}/{article.id}.png"],
                    "submitted_image": ""
                }
            else:
                # Id already exists — just append the new image
                output_data[article.id]["images"].append(f"{folder_path}/{article.id}.png")
    end = time.time()
    print(f"Z-Turbo Total Time: {end - start:.4f} seconds")
    print(f"\tAvg Time for {len(articles)} articles: {(end - start)/len(articles):.4f} seconds")
    del generator



def image_gen_qwen_sdnq(
        articles: List[Article_26],
        used_prompt: str,
        attempt: int,
        output_data:  dict = None
):
    # ── Load ──────────────────────────────────────────────────────────────────────
    generator = QwenImageGenerator(
        use_quantized_matmul=True,  # set False if SDNQ / triton is not installed
        cpu_offload=True,
    )
    generator.load()

    negative_prompt = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"

    # negative_prompt = " " # using an empty string if you do not have specific concept to remove
    # output_data = []
    start = time.time()
    for i, article in enumerate(articles):
        if not article.is_complete:

            prompt = article.prompts[used_prompt][attempt].text
            print(f"{article.id} / {prompt}")
            # ── Generate ──────────────────────────────────────────────────────────────────
            image = generator.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                aspect_ratio="16:9",
                num_inference_steps=20,
                seed=42,
            )

            p = Path(f"{config.RUN_ID}/{MODEL_ID_LLM.split('/')[-1]}/Qwen-Image-SDNQ-uint4-svd-r32/{used_prompt}/{attempt}")
            p.mkdir(parents=True, exist_ok=True)
            folder_path = str(p)
            image.save(f"{folder_path}/{article.id}.png")
            article.prompts[used_prompt][attempt].add_image(f"{folder_path}/{article.id}.png")

            if article.id not in output_data:
                # First time seeing this id — create the entry
                output_data[article.id] = {
                    "id": article.id,
                    "title": article.title,
                    "images": [f"{folder_path}/{article.id}.png"],
                    "submitted_image": ""
                }
            else:
                # Id already exists — just append the new image
                output_data[article.id]["images"].append(f"{folder_path}/{article.id}.png")
    end = time.time()
    print(f"Qwen sdnq Total Time: {end - start:.4f} seconds")
    print(f"\tAvg Time for {len(articles)} articles: {(end - start)/len(articles):.4f} seconds")
    del generator


def image_gen_qwen_2512(
        articles: List[Article_26],
        used_prompt: str,
        attempt: int,
        output_data:  dict = None
):
    # ── Load ──────────────────────────────────────────────────────────────────────
    generator = QwenImage2512Generator(cpu_offload=True)
    generator.load()

    negative_prompt = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"

    # negative_prompt = " " # using an empty string if you do not have specific concept to remove
    # output_data = []
    start = time.time()
    for i, article in enumerate(articles):
        if not article.is_complete:
            prompt = article.prompts[used_prompt][attempt].text
            print(f"{article.id} / {prompt}")
            # ── Generate ──────────────────────────────────────────────────────────────────
            image = generator.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                aspect_ratio="16:9",
                num_inference_steps=20,
                seed=42,
            )

            p = Path(f"{config.RUN_ID}/{MODEL_ID_LLM.split('/')[-1]}/Qwen-Image-2512/{used_prompt}/{attempt}")
            p.mkdir(parents=True, exist_ok=True)
            folder_path = str(p)
            image.save(f"{folder_path}/{article.id}.png")
            article.prompts[used_prompt][attempt].add_image(f"{folder_path}/{article.id}.png")

            if article.id not in output_data:
                # First time seeing this id — create the entry
                output_data[article.id] = {
                    "id": article.id,
                    "title": article.title,
                    "images": [f"{folder_path}/{article.id}.png"],
                    "submitted_image": ""
                }
            else:
                # Id already exists — just append the new image
                output_data[article.id]["images"].append(f"{folder_path}/{article.id}.png")
    end = time.time()
    print(f"Qwen 2512 Total Time: {end - start:.4f} seconds")
    print(f"\tAvg Time for {len(articles)} articles: {(end - start)/len(articles):.4f} seconds")
    del generator


def image_gen_flux2(
        articles: List[Article_26],
        used_prompt: str,
        attempt: int,
        output_data:  dict = None
):
    # ── Load ──────────────────────────────────────────────────────────────────────
    generator = Flux2KleinGenerator(
        device="cuda",
        cpu_offload=True,
    )
    generator.load()

    negative_prompt = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"

    # negative_prompt = " " # using an empty string if you do not have specific concept to remove
    # output_data = []
    start = time.time()
    for i, article in enumerate(articles):
        if not article.is_complete:
            prompt = article.prompts[used_prompt][attempt].text
            print(f"{article.id} / {prompt}")
            # ── Generate ──────────────────────────────────────────────────────────────────
            image = generator.generate(
                prompt=prompt,
                aspect_ratio="16:9",
                num_inference_steps=4,
                seed=0,
            )

            p = Path(f"{config.RUN_ID}/{MODEL_ID_LLM.split('/')[-1]}/FLUX.2-klein-9B/{used_prompt}/{attempt}")
            p.mkdir(parents=True, exist_ok=True)
            folder_path = str(p)
            image.save(f"{folder_path}/{article.id}.png")
            article.prompts[used_prompt][attempt].add_image(f"{folder_path}/{article.id}.png")

            if article.id not in output_data:
                # First time seeing this id — create the entry
                output_data[article.id] = {
                    "id": article.id,
                    "title": article.title,
                    "images": [f"{folder_path}/{article.id}.png"],
                    "submitted_image": ""
                }
            else:
                # Id already exists — just append the new image
                output_data[article.id]["images"].append(f"{folder_path}/{article.id}.png")
    end = time.time()
    print(f"Flux/2 Total Time: {end - start:.4f} seconds")
    print(f"\tAvg Time for {len(articles)} articles: {(end - start)/len(articles):.4f} seconds")
    del generator


# ──────────────────────────────────────────
# CONFIG — swap model_id to try the other
# ──────────────────────────────────────────
MODEL_ID_LLM = config.MODEL_ID_LLM
used_prompts = ["prompt_1"]
prompts = {
    "prompt_1": "Using this article title create a prompt for an image generation model to create a relative to the article image. Return ONLY the prompts. "
                # "Do NOT show your reasoning or thinking process. Do NOT explain steps."
                "Stress that the image should be non-photorealistic"
                "",
    "prompt_2": "Using this article title create a prompt for an image generation model to create a relative to the article image. Return ONLY the prompts. "
                # "Do NOT show your reasoning or thinking process. Do NOT explain steps."
                "Stress that the image should be non-photorealistic"
                "",}

# Read without header
df = pd.read_csv(config.csv_file, encoding="ISO-8859-1")
# Assign proper column names
df.columns = ["article_id", "article_url", "article_title", "image_url"]

articles = []
for _, row in df.iterrows():
    if row["article_id"] == "article_id":
        continue
    article = Article_26(
        id=int(row["article_id"]),
        title=row["article_title"],
        prompts={},
        submitted_image=""
    )
    articles.append(article)
print()

if config.RERUN_INITIAL_PROMPT:
    # ──────────────────────────────────────────
    # LOAD MODEL
    # ──────────────────────────────────────────
    print(f"Loading {MODEL_ID_LLM} … (first run downloads the weights, ~16 GB)")
    # ── Load ──────────────────────────────────────────────────────────────────────
    assistant = Qwen3VLAssistant()
    assistant.load()
    print("Model loaded.\n")



    for used_prompt in used_prompts:
        print(used_prompt)
        for i, article in tqdm(enumerate(articles)):
            context = f"""
            {article.title}
            """

        ## prompt#1: Using this article title create a prompt for an image generation model to create a relative to the article image
            question = (prompts[used_prompt])

            # answer = ask_about_text(context, question)
            answer = assistant.ask_about_text(context=context, question=question, max_new_tokens=512, temperature=0.3, top_p=0.9,)
            # print(answer)
            # articles[i].prompt[used_prompt] = answer
            articles[i].prompts[used_prompt] = [PromptEntry(text=answer)]
        print()

        save_f = f"{config.csv_file[:-4]}_{MODEL_ID_LLM.split('/')[-1]}"
        # df.to_csv(f"{save_f}.csv", index=False)
        with open(f"{save_f}.pickle", "wb") as f:
            pickle.dump(articles, f)
    del assistant
else:
    save_f = f"{config.csv_file[:-4]}_{MODEL_ID_LLM.split('/')[-1]}"
    with open(f"{save_f}.pickle", "rb") as f:
        articles = pickle.load(f)

# articles = articles[:10]
attempts = 0
output_data = {}
while attempts < config.ITERATIONS:

    if is_articles_complete(articles):
        break

    used_prompt = attempts
    # STEP 2. Generate images

    # ── 2.1. Load  the  Z-Image-Turbo pipeline
    if 'zt' in config.IMG_GEN_MODELS:
        print(f"{attempts} / {config.ITERATIONS} gen_zturbo")
        image_gen_zturbo(articles, used_prompts[0], attempts, output_data)
    # ── 2.2. Load  the  QWEN-sdnq pipeline
    if 'sdnq' in config.IMG_GEN_MODELS:
        print(f"{attempts} / { config.ITERATIONS} gen_qwen_sdnq")
        image_gen_qwen_sdnq(articles, used_prompts[0], attempts, output_data)
    # ── 2.3. Load  the  QWEN-2515 pipeline
    if '2512' in config.IMG_GEN_MODELS:
        print(f"{attempts} / { config.ITERATIONS} gen_qwen_2512")
        image_gen_qwen_2512(articles, used_prompts[0], attempts, output_data)
    # ── 2.4. Load  the  flux.2 pipeline
    if 'flux' in config.IMG_GEN_MODELS:
        print(f"{attempts} / { config.ITERATIONS} gen_flux2")
        image_gen_flux2(articles, used_prompts[0], attempts, output_data)

    # STEP 3. CRITIC evaluates all images
    critic(articles, used_prompts[0], attempts)

    # STEP 4. REFINEMENT
    # Actor takes the Critic's notes and rewrites the prompt
    print()
    refinement(articles, used_prompts[0], attempts)
    print()
    attempts +=1

    # Save json
    json_path = Path(f"{config.RUN_ID}/{MODEL_ID_LLM.split('/')[-1]}/articles.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(list(output_data.values()), f, indent=2, ensure_ascii=False)
    # with open(json_path, "w", encoding="utf-8") as f:
    #     json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON to {json_path}")

    save_f = f"{config.RUN_ID}/{config.csv_file.split('/')[-1][:-4]}_{MODEL_ID_LLM.split('/')[-1]}"
    with open(f"{save_f}.pickle", "wb") as f:
        pickle.dump(articles, f)

    with open(json_path, "w", encoding="utf-8") as f:
        f.write(articles_to_json(articles, indent=2))

print()
save_f = f"{config.RUN_ID}/{config.csv_file.split('/')[-1][:-4]}_{MODEL_ID_LLM.split('/')[-1]}"
with open(f"{save_f}.pickle", "wb") as f:
    pickle.dump(articles, f)
#
json_path = Path(f"{config.RUN_ID}/{MODEL_ID_LLM.split('/')[-1]}/articles.json")
with open(json_path, "w", encoding="utf-8") as f:
    f.write(articles_to_json(articles, indent=2))

# output_data = []
for i, article in enumerate(articles):

    selected_img_path = ""
    max_score = 0
    for prompt in article.prompts[used_prompts[0]]:
        if len(prompt.scores) == 0:
            continue
        max_index, max_value = max(enumerate(prompt.scores), key=lambda x: x[1])
        print(prompt.scores)
        if max_value >= max_score:
            max_score = max_value
            selected_img_path = prompt.image_paths[max_index]
    print(selected_img_path)
    p = Path(f"{config.RUN_ID}/{MODEL_ID_LLM.split('/')[-1]}/Final_subm/{used_prompts[0]}/")
    p.mkdir(parents=True, exist_ok=True)
    folder_path = str(p)
    save_path = f"{folder_path}/{article.id}.png"
    shutil.copy(selected_img_path, save_path)
    article.submitted_image = f"{folder_path}/{article.id}.png"


print()
save_f = f"{config.RUN_ID}/{config.csv_file.split('/')[-1][:-4]}_{MODEL_ID_LLM.split('/')[-1]}"
with open(f"{save_f}.pickle", "wb") as f:
    pickle.dump(articles, f)

# Save JSON after the loop
json_path = Path(f"{config.RUN_ID}/articles.json")
with open(json_path, "w", encoding="utf-8") as f:
    f.write(articles_to_json(articles, indent=2))
print(f"Saved JSON to {json_path}")