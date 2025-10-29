import json
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

import faiss

ArrayLike = Union[List[Any], np.ndarray, torch.Tensor]


class CandidateGenerator(ABC):
    """
    Abstract class for candidate generators.

    Given query, generate candidate images and their captions:
    - The pool of candidates (faiss index to query, or local path to images) should be provided in the `__init__` method.
    - Custom method for generating candidates should be called in the `generate_candidates` method.
    - Output should be a tuple of at least two arrays of the same length:
        - Image ids/paths to the candidates in the pool.
        - Similarity scores between the query and the candidates.
    """

    @abstractmethod
    def generate_candidates(
        self,
        queries: ArrayLike,
        num_candidates: int,
        **kwargs,
    ) -> Tuple[ArrayLike, ArrayLike]:
        pass


class EmbeddingCandidateGenerator(CandidateGenerator):
    """
    Candidate generation based on similarity search with image and query embeddings.

    Args:
        image_embeddings: image embeddings, shape: [num_images, embedding_dim]
        image_paths: image paths, shape: [num_images]
    """
    def __init__(
        self,
        image_embeddings: ArrayLike,
        image_paths: ArrayLike,
    ):
        self.image_embeddings = image_embeddings
        if isinstance(image_paths, List):
            self.image_paths = np.array(image_paths)
        else:
            self.image_paths = image_paths

    def generate_candidates(
        self,
        queries: ArrayLike,
        num_candidates: int,
        normalize: bool = True,
        blip2: bool = False,
        **kwargs,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Classical similarity-based retrieval with image and query embeddings.

        Args:
            queries: query embeddings, shape: [num_queries, embedding_dim]
            num_candidates: number of candidates to generate
            normalize: flag to normalize query and image embeddings
            blip2: blip2-based retrieval is a bit different: 
                there are multiple Q-Former channels for each image embedding
        
        Returns:
            sorted_indices: sorted indices of the candidates, shape: [num_queries, num_candidates]
            top_k_similarity_scores: top-k similarity scores, shape: [num_queries, num_candidates]
        """
        if not isinstance(queries, torch.Tensor):
            try:
                queries = torch.tensor(queries)
            except Exception as e:
                raise ValueError(f"Cannot convert queries to torch.Tensor: {e}") from e

        assert queries.ndim == 2, \
            "Queries must be a 2D tensor"

        # Similarity matrix: [num_queries, num_images]
        logits_per_text = EmbeddingCandidateGenerator.retrieval_query_image_embeddings(
            queries,
            self.image_embeddings,
            normalize,
            blip2,
        )

        sorted_values, sorted_indices = torch.sort(logits_per_text, dim=1, descending=True)

        top_k_similarity_scores = sorted_values[:, :num_candidates]
        top_k_indices = sorted_indices[:, :num_candidates]

        return top_k_indices, top_k_similarity_scores

    @staticmethod
    def retrieval_query_image_embeddings(
        query_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
        normalize: bool = True,
        blip2: bool = False,
    ) -> torch.Tensor:
        """
        Compute similarity matrix between queries and images

        Args:
            query_embeddings: initial query embeddings
            image_embeddings: image embeddings
            blip2: blip2-based retrieval is a bit different: 
                there are multiple Q-Former channels for each image embedding

        Returns:
            logits_per_text: similarity matrix between queries and images
                rows: queries
                columns: images
        """
        assert query_embeddings.shape[0] == image_embeddings.shape[0], \
            "Number of queries and images must be the same"
        
        assert query_embeddings.shape[-1] == image_embeddings.shape[-1], \
            "Dimension of query and image embeddings must be the same"

        if normalize:
            query_embeddings = F.normalize(query_embeddings, dim=1)
            image_embeddings = F.normalize(image_embeddings, dim=1)

        # Compute similarity matrix between queries and images
        logits_per_image = torch.matmul(image_embeddings, query_embeddings.t())
        # Shape of image embeddings in BLIP-2: [num_images, n_q, dim] -- Q-former returns image representation as n_q tokens
        # Select the most relevant image token (from n_q) for each query
        if blip2:
            logits_per_image, _ = logits_per_image.max(dim=1) # [num_images, num_queries]

        logits_per_text = logits_per_image.t() # [num_queries, num_images]
        return logits_per_text


class FaissCandidateGenerator(CandidateGenerator):
    """
    Candidate generation based on similarity search with FAISS index.
    """
    def __init__(
            self,
            faiss_index_path: str,
            image_paths_file: str,
        ):
        self.faiss_index = self._load_faiss_index(faiss_index_path)
        self.image_paths = self._load_image_paths(image_paths_file)

    def _load_faiss_index(self, faiss_index_path: str):
        """Load FAISS index from file."""
        print(f"Loading FAISS index from {faiss_index_path}")
        index = faiss.read_index(faiss_index_path)
        print(f"Loaded {index.ntotal} vectors with dimension {index.d}")
        return index

    def _load_image_paths(self, paths_file: str) -> List[str]:
        """Load image paths corresponding to FAISS index."""
        with open(paths_file, 'r') as f:
            return np.array([line.strip() for line in f if line.strip()])
    
    def generate_candidates(
        self,
        queries: np.ndarray,
        num_candidates: int,
        normalize: bool = True,
        **kwargs,
    ) -> Tuple[ArrayLike, ArrayLike]:
        assert queries.shape[1] == self.faiss_index.d, \
            f"Dimension mismatch: {queries.shape[1]} != {self.faiss_index.d}"
        
        if normalize:
            faiss.normalize_L2(queries)

        # Compute similarity matrix between queries and images
        distances, indices = self.faiss_index.search(queries, num_candidates)
        return indices, distances
