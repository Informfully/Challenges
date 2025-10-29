from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LLMWrapper(ABC):
    pipeline: Optional[Any] = field(default=None)

    def __post_init__(self):
        if self.pipeline is None:
            raise ValueError("Pipeline must be provided")

    @abstractmethod
    def get_prompt(self, *args, **kwargs):
        pass

    @abstractmethod
    def generate(self, *args, **kwargs):
        pass