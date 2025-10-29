import argparse
import json
import os

import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.newsimages import NewsImageDataset
from src.models.llama import Llama38BWrapper
from src.utils.utils import yaml_to_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--llm_config_path", type=str, required=True)
    parser.add_argument("--system_prompt_path", type=str, required=True)
    parser.add_argument("--user_prompt_path", type=str, required=True)
    parser.add_argument("--num_captions", type=int, default=10)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--debug", action="store_true", default=False, help="Debug mode (10 samples)")
    return parser.parse_args()


def configure_flash_attention():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        major, minor = torch.cuda.get_device_capability(device)
        gpu_name = torch.cuda.get_device_name(device)
        print(f"GPU: {gpu_name}, Compute Capability: {major}.{minor}")

        # Flash attention requires major >= 8
        if major >= 8:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        else:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
    else:
        print("CUDA not available, running on CPU.")


def generate_captions_from_titles(args):
    dataset = NewsImageDataset(
        csv_path=args.csv_path,
        get_images=False,
    )

    system_prompt = yaml_to_dict(args.system_prompt_path)["prompt"]
    user_prompt_template = yaml_to_dict(args.user_prompt_path)["prompt"]

    llm_config = yaml_to_dict(args.llm_config_path)
    model_id = llm_config["model_id"]
    generation_config = llm_config["generation_config"]

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    LLM = Llama38BWrapper(pipeline=pipeline)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # For now, we use a batch size of 1 for generation for computational reasons
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = []

    for idx, batch in tqdm(enumerate(dataloader), total=len(dataset)):
        title = batch["title"]
        id_ = batch["article_id"].cpu().item()
        url = batch["url"]

        user_prompt = user_prompt_template.format(num_captions=args.num_captions, title=title)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            prompt = LLM.get_prompt(messages)
            captions = LLM.generate(prompt, generation_config=generation_config)
        except:
            print(f"Error generating captions for:\n{messages}")
            continue
        
        captions = captions.split(";")
        for i, caption in enumerate(captions):
            if "\n" in caption:
                captions[i] = caption.split("\n")[-1]
        
        captions = [caption.strip() for caption in captions if caption]

        results.append({
            "article_id": id_,
            "title": title,
            "captions": captions,
            "article_url": url,
        })

        if args.debug and idx > 10:
            break
    
    print(f"Generated {len(results)} captions for {len(dataset)} articles")

    with open(args.output_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    configure_flash_attention()
    args = parse_args()
    generate_captions_from_titles(args)