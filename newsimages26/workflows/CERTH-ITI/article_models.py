# article_models.py

from dataclasses import dataclass, field
from typing import Dict, List
import json


@dataclass
class PromptEntry:
    text: str
    image_paths: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)

    def add_image(self, image_path: str) -> None:
        """Ingest a single image path."""
        if self.scores:
            raise RuntimeError("Cannot add images after scores have been assigned.")
        self.image_paths.append(image_path)

    def add_images(self, image_paths: List[str]) -> None:
        """Ingest multiple image paths at once."""
        for path in image_paths:
            self.add_image(path)

    def set_scores(self, scores: List[float]) -> None:
        """Assign scores once all images have been ingested."""
        if not self.image_paths:
            raise RuntimeError("No images have been added yet.")
        if len(scores) != len(self.image_paths):
            raise ValueError(
                f"Expected {len(self.image_paths)} scores, got {len(scores)}."
            )
        self.scores = scores

    def add_score(self, score: float) -> None:
        """Append a single score. Cannot exceed the number of ingested images."""
        if not self.image_paths:
            raise RuntimeError("No images have been added yet.")
        if len(self.scores) >= len(self.image_paths):
            raise RuntimeError(
                f"All {len(self.image_paths)} images already have a score."
            )
        self.scores.append(score)

    @property
    def is_complete(self) -> bool:
        """True when every image has a corresponding score."""
        return bool(self.image_paths) and len(self.scores) == len(self.image_paths)

    @property
    def average_score(self) -> float | None:
        return sum(self.scores) / len(self.scores) if self.scores else None


@dataclass
class Article_26:
    id: int
    title: str
    submitted_image: str
    prompts: Dict[str, List[PromptEntry]] = field(default_factory=dict)
    is_complete: bool = field(default=False)




def articles_to_json(articles: List[Article_26], indent: int = 2) -> str:
    """Serialize a list of Article_26 objects to a JSON string."""

    def serialize_prompt_entry(entry: PromptEntry) -> dict:
        return {
            "text": entry.text,
            "image_paths": entry.image_paths,
            "scores": entry.scores,
            "is_complete": entry.is_complete,
            "average_score": entry.average_score,
        }

    def serialize_article(article: Article_26) -> dict:
        return {
            "id": article.id,
            "title": article.title,
            "is_complete": article.is_complete,
            "submitted_image": article.submitted_image,
            "prompts": {
                model_name: [serialize_prompt_entry(e) for e in entries]
                for model_name, entries in article.prompts.items()
            },
        }

    return json.dumps([serialize_article(a) for a in articles], indent=indent, ensure_ascii=False)
