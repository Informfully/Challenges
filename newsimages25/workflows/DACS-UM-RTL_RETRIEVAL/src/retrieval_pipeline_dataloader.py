import argparse
import json
import os
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.datasets.newsimages import NewsImageDataset, NewsImagesCollator, NewsImagesDataHandler
from src.models.aesthetic_model import AestheticModel
from src.models.candidate_generators import EmbeddingCandidateGenerator
from src.models.configs import get_vlm_wrapper
from src.models.reranking import get_reranking_algorithm
from src.utils.metrics import RetrievalEvaluator
from src.utils.utils import yaml_to_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)

    parser.add_argument("--model_family", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)

    parser.add_argument(
        "--data_split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--subset_split",
        type=str,
        default=None,
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--log_retrieved_images", type=str, default="./results/development")
    parser.add_argument("--num_candidates", type=int, default=100)

    parser.add_argument(
        "--reranking_algorithm",
        type=str,
        default="identity",
        choices=["identity", "vlm_judge", "llm_rewriting", "aesthetics"],
    )
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
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=28)

    return parser.parse_args()


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    experiment_id = f"{args.model_family}_{args.model_id}_{uuid.uuid4().hex[:8]}"

    if not args.debug:
        wandb.init(
            project=f"newsimages-development-data-{args.data_split}",
            name=experiment_id,
            config=args,
        )

    data_handler = NewsImagesDataHandler(
        csv_path=args.csv_path,
        image_path=args.image_path,
    )

    if args.subset_split is not None:
        assert args.subset_split is not None, "Subset csv is required for subset split"
        dataset = data_handler.get_subset(args.subset_split)
    else:
        train_data, val_data, test_data = data_handler.split_csv_train_val_test(
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=args.seed,
        )
        dataset = {"train": train_data, "val": val_data, "test": test_data}.get(args.data_split, None)
        assert dataset is not None, f"Unknown data split: {args.data_split}"

    dataset = NewsImageDataset(
        dataset_df=dataset,
        image_path=args.image_path,
    )

    wrapper = get_vlm_wrapper(
        model_family=args.model_family,
        model_id=args.model_id,
        device=device,
    )

    collator = NewsImagesCollator(
        wrapper=wrapper,
        process_images=True,
        process_titles=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    image_embeddings = []
    text_embeddings = []
    query_idx = []
    image_paths = []
    titles = []
    article_ids = []
    
    wrapper.model.eval()

    # region get embeddings
    # Pass through the dataset to get image and text embeddings, and related metadata
    for i, batch in tqdm(enumerate(dataloader), desc="Computing image and title embeddings"):
        text_inputs = {
            "input_ids": batch["input_ids"],
        }
        if "attention_mask" in batch:
            text_inputs["attention_mask"] = batch["attention_mask"]

        image_inputs = {
            "pixel_values": batch["pixel_values"],
        }

        with torch.no_grad():
            text_embeds = wrapper.get_text_embeddings(inputs=text_inputs)
            image_embeds = wrapper.get_image_embeddings(inputs=image_inputs)

        image_embeddings.append(image_embeds.detach().cpu())
        text_embeddings.append(text_embeds.detach().cpu())
        image_paths.extend(batch["image_paths"])
        titles.extend(batch["titles"])
        article_ids.extend(batch["article_id"])
        offset = i * args.batch_size
        query_idx.extend([offset + j for j in range(len(batch["pixel_values"]))])
    
    image_embeddings = torch.cat(image_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    
    query_idx = np.array(query_idx)
    image_paths = np.array(image_paths)
    titles = np.array(titles)
    article_ids = np.array(article_ids)
    #endregion get embeddings


    #region Initial retrieval (candidate generation)
    retriever = EmbeddingCandidateGenerator(
        image_embeddings=image_embeddings,
        image_paths=image_paths,
    )

    top_k_indices, top_k_similarity_scores = retriever.generate_candidates(
        queries=text_embeddings,
        num_candidates=args.num_candidates,
        normalize=True,
        blip2=True if "blip2" in args.model_family else False,
    )
    top_k_image_paths = image_paths[top_k_indices]
    
    # Log initial retrieval results 
    retrieved_images_topk = []
    for i in range(len(titles)):
        retrieved_images_topk.append({
            "query": titles[i],
            "ground_truth": image_paths[i],
            "retrieved": top_k_image_paths[i].tolist(),
            "similarity_scores": top_k_similarity_scores[i].tolist(),
        })
    retrieved_images_path = os.path.join(args.log_retrieved_images, "initial_retrieval.json")
    os.makedirs(os.path.dirname(retrieved_images_path), exist_ok=True)
    with open(retrieved_images_path, "w") as f:
        json.dump(retrieved_images_topk, f)

    # Make sure that top_k is not greater than the number of candidates
    at_k = [1, 5, 10, 25, 50, 100]
    top_k = []
    for k in at_k:
        if k > args.num_candidates:
            break
        top_k.append(k)

    evaluator = RetrievalEvaluator(
        top_k=top_k,
        metrics=['hits', 'mrr'],
    )

    metrics, _ = evaluator.evaluate(top_k_indices, query_idx)

    print("Initial retrieval metrics:")
    for metric in metrics:
        print(f"{metric}: {round(metrics[metric], 3)}")

    print("-" * 100)

    if not args.debug:
        wandb.log({
            "initial_retrieval": metrics,
        })
    #endregion Initial retrieval (candidate generation)

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
        reranking_kwargs = {
            "llm_captions_path": args.llm_captions_path,
            "weight_reranking": args.reranking_llm_weight,
            "vlm_wrapper": wrapper,
            "num_captions": args.reranking_llm_num_captions,
        }

        rerank_forward_kwargs = {
            "top_k_image_embeddings": image_embeddings[top_k_indices],
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
    reranked_image_paths = image_paths[reranked_indices]

    # Log reranking results 
    # TODO: Think of local logging class that saves results locally to avoid repeating the same code
    #   However, different loggers might need to log different information (e.g. synthetic captions, scores, etc.)
    retrieved_images_topk = []
    for i in range(len(titles)):
        retrieved_images_topk.append({
            "query": titles[i],
            "ground_truth": image_paths[i],
            "retrieved": reranked_image_paths[i].tolist(),
            "similarity_scores": reranked_similarity_scores[i].tolist(),
        })
    retrieved_images_path = os.path.join(args.log_retrieved_images, "reranking.json")
    os.makedirs(os.path.dirname(retrieved_images_path), exist_ok=True)
    with open(retrieved_images_path, "w") as f:
        json.dump(retrieved_images_topk, f)

    reranking_evaluator = RetrievalEvaluator(
        top_k=top_k,
        metrics=['hits', 'mrr'],
    )

    reranking_metrics, _ = reranking_evaluator.evaluate(reranked_indices, query_idx)

    delta_metrics = {}
    print("Reranking metrics:")
    for metric in reranking_metrics:
        print(f"{metric}: {round(reranking_metrics[metric], 3)}")
        delta_metrics[metric] = reranking_metrics[metric] - metrics[metric]
    print("-" * 100)

    if not args.debug:
        wandb.log({
            "reranking": reranking_metrics,
            "delta": delta_metrics,
        })
    #endregion Re-ranking

    if not args.debug:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
