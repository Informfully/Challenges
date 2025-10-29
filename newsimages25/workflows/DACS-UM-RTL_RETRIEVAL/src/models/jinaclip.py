from dataclasses import dataclass, field
from typing import Any, Dict
import torch
from transformers import AutoProcessor, AutoModel
from src.models.vlm_wrapper import VLMWrapperRetrieval

class JinaWrapper(VLMWrapperRetrieval):
    model: Any = field(
        default_factory=lambda: AutoModel.from_pretrained(
            'jinaai/jina-clip-v2', trust_remote_code=True, device_map={"": 0}, torch_dtype=torch.float16
        )
    )

    processor: Any = field(
        default_factory=lambda: AutoProcessor.from_pretrained(
            'jinaai/jina-clip-v2', trust_remote_code=True
        )
    )
   
    def process_inputs(self, images=None, text=None) -> Dict[str, Any]:   
        assert images is not None or text is not None
        return self.processor(
            images=images,
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

    def get_embeddings(self, inputs, **kwargs):
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        return outputs
    
    def get_text_embeddings(self, inputs: Dict[str, Any], **kwargs) -> Any:
        with torch.no_grad():
            return self.model.get_text_features(
                **inputs
            )
    def get_image_embeddings(self, inputs: Dict[str, Any], **kwargs) -> Any:
        with torch.no_grad():
            return self.model.get_image_features(pixel_values=inputs["pixel_values"])