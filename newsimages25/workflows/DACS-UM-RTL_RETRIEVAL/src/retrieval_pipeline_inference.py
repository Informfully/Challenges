import argparse
import json
import os
import uuid

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.datasets.newsimages import NewsImageDataset, NewsImagesCollator, NewsImagesDataHandler
from src.models.aesthetic_model import AestheticModel
from src.models.candidate_generators import FaissCandidateGenerator
from src.models.configs import get_vlm_wrapper
from src.models.reranking import get_reranking_algorithm
from src.utils.utils import yaml_to_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--faiss_index_path",
        type=str,
        required=True,
        help="Path to the FAISS index file"
    )
    parser.add_argument(
        "--image_paths_file",
        type=str,
        required=True,
        help="Path to txt file containing image paths corresponding to FAISS index"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to CSV file for queries dataset"
    )
    parser.add_argument(
        "--subset_split",
        type=str,
        required=True,
        help="Subset split of queries dataset"
    )
    parser.add_argument("--model_family", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    
    parser.add_argument("--output_path", type=str, default="./results/inference/last")
    parser.add_argument("--num_candidates", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    
    parser.add_argument("--reranking_algorithm", type=str, default="identity")
    parser.add_argument(
        "--reranking_config",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--reranking_vlm_prompt_path",
        type=str,
        default="configs/vlm_reranking/prompt.txt",
    )
    parser.add_argument("--reranking_vlm_weight", type=float, default=0.2)
    parser.add_argument("--reranking_llm_weight", type=float, default=0.5)
    parser.add_argument("--reranking_llm_num_captions", type=int, default=5)

    parser.add_argument("--reranking_aesthetic_local_path", type=str, default=None)
    parser.add_argument("--reranking_aesthetic_model_name", type=str, default="ViT-L-14")
    parser.add_argument("--reranking_aesthetic_weight", type=float, default=0.2)
    parser.add_argument("--llm_captions_path", type=str, default=None)

    return parser.parse_args()


def main(args):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_id = f"inference_{args.model_family}_{args.model_id}_{uuid.uuid4().hex[:8]}"

    data_handler = NewsImagesDataHandler(
        csv_path=args.csv_path,
    )

    # TODO: to be updated with the articles sampled for the evaluation
    if args.subset_split is not None:
        test_data = data_handler.get_subset(args.subset_split)
    else:
        _, _, test_data= data_handler.split_csv_train_val_test()
    
    query_dataset = NewsImageDataset(
        dataset_df=test_data,
        get_images=False,
    )
    
    wrapper = get_vlm_wrapper(
        model_family=args.model_family,
        model_id=args.model_id,
        device=device,
    )
    wrapper.model.eval()
    
    collator = NewsImagesCollator(
        wrapper=wrapper,
        process_images=False,
        process_titles=True,
    )

    query_dataloader = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    query_embeddings = []
    titles = []
    article_ids = []
    
    wrapper.model.eval()
    
    for batch in tqdm(query_dataloader, desc="Computing title embeddings"):
        text_inputs = {
            "input_ids": batch["input_ids"],
        }
        if "attention_mask" in batch:
            text_inputs["attention_mask"] = batch["attention_mask"]

        with torch.no_grad():
            batch_embeds = wrapper.get_text_embeddings(inputs=text_inputs)
            query_embeddings.append(batch_embeds.detach().cpu())
            titles.extend(batch["titles"])
            article_ids.extend(batch["article_id"])
    
    query_embeddings = torch.cat(query_embeddings, dim=0)
    query_embeddings_np = query_embeddings.numpy()
    article_ids = np.array(article_ids)
    
    candidate_generator = FaissCandidateGenerator(
        faiss_index_path=args.faiss_index_path,
        image_paths_file=args.image_paths_file,
    )
    
    top_k_indices, top_k_similarity_scores = candidate_generator.generate_candidates(
        queries=query_embeddings_np,
        num_candidates=args.num_candidates,
        normalize=True,
    )
    top_k_image_paths = candidate_generator.image_paths[top_k_indices]

    #region Re-ranking
    if args.reranking_config is not None:
        reranking_config = yaml_to_dict(args.reranking_config)
    else:
        reranking_config = {}

    if args.reranking_algorithm == "vlm_judge":
        assert "model_family" in reranking_config, "model_family is required for VLM Judge"
        assert "model_id" in reranking_config, "model_id is required for VLM Judge"

        vlm_judge_wrapper = get_vlm_wrapper(
            model_family=reranking_config["model_family"],
            model_id=reranking_config["model_id"],
            device=reranking_config.get("device", device),
        )
        reranking_kwargs = {
            "vlm_wrapper": vlm_judge_wrapper,
            "prompt_path": args.reranking_vlm_prompt_path,
            "weight_reranking": args.reranking_vlm_weight,
        }
        rerank_forward_kwargs = {
            "titles": titles,
            "top_k_image_paths": top_k_image_paths,
        }
    elif args.reranking_algorithm == "llm_rewriting":
        image_embeddings = []
        for image_paths_query in tqdm(top_k_image_paths, desc="Computing image embeddings"):
            image_embeddings_candidates = []
            for image_path in image_paths_query:
                image = [Image.open(image_path).convert("RGB")]
                image_inputs = {
                    "pixel_values": wrapper.process_inputs(images=image)["pixel_values"],
                }
                with torch.no_grad():
                    batch_embeds = wrapper.get_image_embeddings(inputs=image_inputs)
                    image_embeddings_candidates.append(batch_embeds.detach().cpu())
            image_embeddings_candidates = torch.cat(image_embeddings_candidates, dim=0)
            image_embeddings.append(image_embeddings_candidates)
        image_embeddings = torch.stack(image_embeddings)
        
        reranking_kwargs = {
            "llm_captions_path": args.llm_captions_path,
            "weight_reranking": args.reranking_llm_weight,
            "vlm_wrapper": wrapper,
            "num_captions": args.reranking_llm_num_captions,
        }

        rerank_forward_kwargs = {
            "top_k_image_embeddings": image_embeddings,
            "article_ids": article_ids,
            "titles": titles,
        }
    elif args.reranking_algorithm == "aesthetics":
        reranking_kwargs = {
            "aesthetic_model": AestheticModel(
                model_name=args.reranking_aesthetic_model_name,
                local_path=args.reranking_aesthetic_local_path,
                device=device,
            ),
            "weight_reranking": args.reranking_aesthetic_weight,
        }
        rerank_forward_kwargs = {
            "top_k_image_paths": top_k_image_paths,
        }
    elif args.reranking_algorithm == "identity":
        reranking_kwargs = {}
        rerank_forward_kwargs = {}
    else:
        raise ValueError(f"Unknown reranking algorithm: {args.reranking_algorithm}")

    reranker = get_reranking_algorithm(args.reranking_algorithm, **reranking_kwargs)

    reranked_indices, reranked_similarity_scores = reranker.rerank(
        sorted_indices=top_k_indices,
        similarity_scores=top_k_similarity_scores,
        **rerank_forward_kwargs,
    )
    reranked_image_paths = top_k_image_paths = candidate_generator.image_paths[reranked_indices]
    
    # Save results
    results = []
    for i, title in enumerate(titles):
        result = {
            "article_id": str(article_ids[i]),
            "query": title,
            "retrieved_images": reranked_image_paths[i].tolist(),
            "similarity_scores": reranked_similarity_scores[i].tolist(),
            "indices": reranked_indices[i].tolist(),
        }
        results.append(result)
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Save detailed results
    output_file = os.path.join(args.output_path, "inference_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f)
    
    # Save summary
    summary = {
        "experiment_id": experiment_id,
        "model_family": args.model_family,
        "model_id": args.model_id,
        "num_queries": len(titles),
        "num_candidates": args.num_candidates,
        "reranking_algorithm": args.reranking_algorithm,
        "index_size": candidate_generator.faiss_index.ntotal,
        "output_file": output_file,
    }
    
    summary_file = os.path.join(args.output_path, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f)


if __name__ == "__main__":
    args = parse_args()
    main(args) 