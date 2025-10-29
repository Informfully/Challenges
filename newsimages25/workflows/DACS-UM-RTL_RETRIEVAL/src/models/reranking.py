import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.models.vlm_wrapper import VLMWrapperGeneration
from src.utils.utils import json_to_dict

ArrayLike = Union[List[Any], np.ndarray, torch.Tensor]


class Reranking(ABC):
    """
    Abstract class for reranking.

    Reranking inputs are sorted indices, similarity scores, and other features depending on the algorithm.
    Returns tuples of two arrays: 
    - reranked indices matrix, shape: [num_queries, num_candidates]
    - updated similarity scores, shape: [num_queries, num_candidates]
    """

    @abstractmethod
    def rerank(self, *args, **kwargs) -> Tuple[ArrayLike, ArrayLike]:
        pass


class IdentityReranking(Reranking):
    """
    Identity reranking. No reranking is performed.
    """

    def rerank(
            self,
            sorted_indices: ArrayLike,
            similarity_scores: ArrayLike,
            **kwargs,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Rerank the sorted indices and similarity scores.
        """
        return sorted_indices, similarity_scores


class RerankingVLMJudge(Reranking):
    """
    Reranking using VLM Judge.
    """
    def __init__(
            self,
            vlm_wrapper: VLMWrapperGeneration,
            prompt_path: str,
            weight_reranking: float = 0.5,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.vlm_wrapper = vlm_wrapper
        self.vlm_wrapper.model.eval()
        self.prompt_path = prompt_path
        self.weight_reranking = weight_reranking
        self.prompt = self.load_prompt(prompt_path)

    def rerank(
            self,
            sorted_indices: ArrayLike,
            similarity_scores: ArrayLike,
            titles: ArrayLike,
            top_k_image_paths: ArrayLike,
            score_only: bool = True,
            **kwargs,
    ) -> Tuple[ArrayLike, ArrayLike]:
        vlm_outputs, scores = self.get_vlm_outputs(sorted_indices, titles, top_k_image_paths, score_only=score_only)

        norm_l1_similarity = similarity_scores.sum(axis=1, keepdims=True)
        normalized_similarity_scores = similarity_scores / norm_l1_similarity

        mean_scores_per_title_candidate = scores.mean(axis=2)
        norm_l1_reranking = mean_scores_per_title_candidate.sum(axis=1, keepdims=True)
        normalized_reranking_scores = mean_scores_per_title_candidate / norm_l1_reranking

        assert normalized_reranking_scores.shape == normalized_similarity_scores.shape

        if isinstance(normalized_similarity_scores, torch.Tensor):
            normalized_similarity_scores = normalized_similarity_scores.float()
        else:
            normalized_similarity_scores = torch.tensor(normalized_similarity_scores)
        if isinstance(normalized_reranking_scores, torch.Tensor):
            normalized_reranking_scores = normalized_reranking_scores.float()
        else:
            normalized_reranking_scores = torch.tensor(normalized_reranking_scores)

        reranked_scores = (1 - self.weight_reranking) * normalized_similarity_scores + \
            self.weight_reranking * normalized_reranking_scores

        if not isinstance(sorted_indices, torch.Tensor):
            sorted_indices = torch.tensor(sorted_indices)

        reranked_score_indices = torch.argsort(reranked_scores, dim=1, descending=True)
        reranked_indices = torch.gather(sorted_indices, 1, reranked_score_indices)

        return reranked_indices, reranked_scores
    
    def get_vlm_outputs(
            self,
            sorted_indices: ArrayLike,
            titles: ArrayLike,
            top_k_image_paths: ArrayLike,
            batch_size: int = 1,
            score_only: bool = False,
    ) -> List[Dict[str, Any]]:
        scores = []
        vlm_outputs = []
        num_criteria = len(self.dimensions)
        num_candidates = len(top_k_image_paths[0])
        
        for i in tqdm(
            range(len(sorted_indices)),
            desc="Processing titles with VLM",
            total=len(sorted_indices),
        ):
            all_generated = []
            title_scores = []
            curr_vlm_outputs = {}
            curr_vlm_outputs["title"] = titles[i]
            curr_vlm_outputs["vlm"] = []
            for j in tqdm(
                range(0, num_candidates, batch_size),
                desc="Processing candidates in batches",
                total=math.ceil(num_candidates / batch_size),
            ):
                decoded_batch = self.judge_with_vlm(
                    top_k_image_paths[i][j: min(j+batch_size, num_candidates)],
                    titles[i],
                    [criterion for criterion in self.dimensions.values()],
                    score_only=score_only,
                )
                all_generated.extend(decoded_batch)
            
            for k, decoded in enumerate(all_generated):
                curr_image_path_idx = k // num_criteria
                try:
                    score, explanation = decoded.split(";;")
                except Exception as e:
                    try:
                        score, explanation = decoded.split(";")
                    except Exception as e:
                        print(f"Failed to parse VLM output: {decoded}")
                        score = 2.5
                        explanation = "NaN"
                curr_vlm_outputs["vlm"].append({
                    "image_path": top_k_image_paths[i][curr_image_path_idx],
                    "score": score,
                    "explanation": explanation,
                })
                title_scores.append(float(score))
            title_scores = np.array(title_scores).reshape(num_candidates, num_criteria)
            vlm_outputs.append(curr_vlm_outputs)
            scores.append(title_scores)
        scores = np.stack(scores)
        return vlm_outputs, scores
    
    def judge_with_vlm(
            self,
            image_path: Union[str, List[str]],
            title: str,
            criterion: Union[str, List[str]],
            max_new_tokens: int = 64,
            score_only: bool = False,
    ) -> Dict[str, Any]:
        if isinstance(criterion, str):
            criterion = [criterion]
        if isinstance(image_path, str):
            image_path = [image_path]
        prompts = [self.prompt.format(title=title, criterion=c) for c in criterion]
        images = [Image.open(image_path).convert("RGB") for image_path in image_path]
        inputs = self.vlm_wrapper.process_inputs(
            images=images,
            prompt=prompts,
        )
        with torch.no_grad():
            generated_ids = self.vlm_wrapper.generate(
                inputs=inputs,
                max_new_tokens=max_new_tokens if not score_only else 5,
                do_sample=False
            )
        decoded = self.vlm_wrapper.decode(generated_ids, input_ids=inputs.input_ids, return_generated_text_only=True)
        return decoded
    
    def load_prompt(self, prompt_path: str) -> str:
        with open(prompt_path, "r") as f:
            return f.read()
    
    @property
    def dimensions(self) -> List[str]:
        return {
            "image_quality": "**Image quality** – Is the image of high quality? 1 - very low quality, 5 - very high quality",
            "trustworthiness": "**Trustworthiness** – Does the image appear to originate from a trustworthy source? 1 - not trustworthy, 5 - very trustworthy",
            "bias": "**Stereotypes/biases free** – To what extent does the image avoid promoting stereotypes or biases related to the article title? 1 - strongly promotes bias, 5 - no signs of bias",
            "clickbait": "**Clickbait-free** – To what extent does the image avoid using clickbait elements? 1 - very clickbait-like, 5 - no signs of clickbait"
        }


class AestheticsReranking(Reranking):
    def __init__(self, aesthetic_model: torch.nn.Module, weight_reranking: int  = 0.1):
        self.aesthetic_model = aesthetic_model
        self.weight_reranking = weight_reranking

    def rerank(self, sorted_indices: ArrayLike, similarity_scores: ArrayLike, top_k_image_paths: ArrayLike, **kwargs):
        aesthetic_scores = self.return_aesthetic_scores(top_k_image_paths)
        norm_l1_similarity = similarity_scores.sum(axis=1, keepdims=True)
        normalized_similarity_scores = similarity_scores / norm_l1_similarity
        normalized_aesthetic_scores = aesthetic_scores/aesthetic_scores.sum(axis=1, keepdims=True)

        if isinstance(normalized_similarity_scores, torch.Tensor):
            normalized_similarity_scores = normalized_similarity_scores.float()
        else:
            normalized_similarity_scores = torch.tensor(normalized_similarity_scores)
        if isinstance(normalized_aesthetic_scores, torch.Tensor):
            normalized_aesthetic_scores = normalized_aesthetic_scores.float()
        else:
            normalized_aesthetic_scores = torch.tensor(normalized_aesthetic_scores)

        reranked_scores = (1 - self.weight_reranking) * normalized_similarity_scores + \
            self.weight_reranking * normalized_aesthetic_scores

        if not isinstance(sorted_indices, torch.Tensor):
            sorted_indices = torch.tensor(sorted_indices)

        reranked_score_indices = torch.argsort(reranked_scores, dim=1, descending=True)
        reranked_indices = torch.gather(sorted_indices, 1, reranked_score_indices)

        return reranked_indices, reranked_scores

    def return_aesthetic_scores(self, top_k_image_paths: ArrayLike):
        """
        Returns aesthetic scores for each image in the top_k_image_paths.
        """
        num_queries = top_k_image_paths.shape[0]
        self.aesthetic_model.eval()
        aesthetic_scores = []
        for query in tqdm(range(num_queries), desc="Aesthetic model inference", total=num_queries):
            images = [Image.open(image) for image in top_k_image_paths[query]]
            prediction = self.aesthetic_model(images)
            aesthetic_scores.append(prediction)
        aesthetic_scores = torch.stack(aesthetic_scores)
        return aesthetic_scores


class RerankingLLMReWriting(Reranking):
    """
    Reranking using LLM rewriting.
    """
    def __init__(
            self, 
            llm_captions_path: str, 
            vlm_wrapper: VLMWrapperGeneration,
            weight_reranking: float = 0.5, 
            num_captions: int = 5,
            batch_size: int = 32,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_captions_path = llm_captions_path
        self.vlm_wrapper = vlm_wrapper
        self.vlm_wrapper.model.eval()
        self.weight_reranking = weight_reranking
        self.llm_captions = self.load_llm_captions(llm_captions_path)
        self.num_captions = num_captions
        self.batch_size = batch_size

    def rerank(
            self, 
            sorted_indices: ArrayLike,
            similarity_scores: ArrayLike,
            top_k_image_embeddings: ArrayLike,
            article_ids: ArrayLike,
            titles: ArrayLike,
            **kwargs,
    ) -> Tuple[ArrayLike, ArrayLike]:
        all_captions = []
        for i in range(len(article_ids)):
            article_id = article_ids[i]
            captions = self.llm_captions[article_id]
            captions_selected = captions[:self.num_captions]
            if len(captions_selected) < self.num_captions:
                captions_selected.extend([""] * (self.num_captions - len(captions_selected)))
            all_captions.extend(captions_selected)
        
        all_captions_embeddings = []
        for i in tqdm(
            range(0, len(all_captions), self.batch_size),
            desc="Processing captions in batches",
            total=math.ceil(len(all_captions) / self.batch_size),
        ):
            batch_captions = all_captions[i: min(i + self.batch_size, len(all_captions))]
            captions_processed = self.vlm_wrapper.process_inputs(text=batch_captions)
            with torch.no_grad():
                caption_embeddings = self.vlm_wrapper.get_text_embeddings(captions_processed)
            all_captions_embeddings.append(caption_embeddings.cpu())
        all_captions_embeddings = torch.vstack(all_captions_embeddings)
        all_captions_embeddings = F.normalize(all_captions_embeddings, dim=1)

        all_captions_embeddings = all_captions_embeddings.reshape(
            len(article_ids),
            self.num_captions,
            -1,
        )

        top_k_image_embeddings = F.normalize(top_k_image_embeddings, dim=-1)

        caption_similarities = torch.matmul(top_k_image_embeddings, all_captions_embeddings.permute(0, 2, 1))

        caption_similarities_max = caption_similarities.max(dim=-1).values

        if isinstance(similarity_scores, torch.Tensor):
            similarity_scores = similarity_scores.float()
        else:
            similarity_scores = torch.tensor(similarity_scores)
        if isinstance(caption_similarities_max, torch.Tensor):
            caption_similarities_max = caption_similarities_max.float()
        else:
            caption_similarities_max = torch.tensor(caption_similarities_max)

        reranked_scores = (1 - self.weight_reranking) * similarity_scores + \
            self.weight_reranking * caption_similarities_max

        if not isinstance(sorted_indices, torch.Tensor):
            sorted_indices = torch.tensor(sorted_indices)

        reranked_score_indices = torch.argsort(reranked_scores, dim=1, descending=True)
        reranked_indices = torch.gather(sorted_indices, 1, reranked_score_indices)

        return reranked_indices, reranked_scores
    
    def load_llm_captions(self, llm_captions_path: str, use_article_id_num: bool = True) -> List[str]:
        captions_all = json_to_dict(llm_captions_path)
        captions_dict = {}
        if not use_article_id_num:
            for caption in captions_all:
                captions_dict[caption["article_id"]] = caption["captions"]
        else:
            for i, caption in enumerate(captions_all):
                captions_dict[i] = caption["captions"]
        return captions_dict


# To add a new reranking algorithm:
# - implement a new class that inherits from Reranking
# - add a new entry to the dictionary.
RERANKING_ALGORITHMS = {
    "identity": {
        "class": IdentityReranking,
    },
    "vlm_judge": {
        "class": RerankingVLMJudge,
    },
    "llm_rewriting": {
        "class": RerankingLLMReWriting,
    },
    "aesthetics": {
        "class": AestheticsReranking
        },
}


def get_reranking_algorithm(algorithm_name: str, **kwargs) -> Reranking:
    """
    Get reranking algorithm by name.
    """
    if algorithm_name not in RERANKING_ALGORITHMS:
        raise ValueError(
            f"Unexpected reranking algorithm: {algorithm_name}. "
            f"Choose from: {list(RERANKING_ALGORITHMS.keys())}"
        )
    
    algorithm_config = RERANKING_ALGORITHMS[algorithm_name]
    algorithm_class = algorithm_config["class"]
    return algorithm_class(**kwargs)