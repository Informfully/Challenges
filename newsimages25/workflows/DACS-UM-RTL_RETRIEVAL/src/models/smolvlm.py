from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.models.vlm_wrapper import VLMWrapperGeneration


@dataclass
class SmolVLMWrapper(VLMWrapperGeneration):
    model: Any = field(
        default_factory=lambda: AutoModelForImageTextToText.from_pretrained(
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",  
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
        )
    )
    processor: Any = field(
        default_factory=lambda: AutoProcessor.from_pretrained(
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" 
        )
    )

    def process_inputs(self, **kwargs):
        required_keys = {'image', 'prompt'}
        if not required_keys.issubset(kwargs.keys()):
            raise ValueError(f"Missing required arguments: {required_keys - set(kwargs.keys())}")
        images = kwargs.get("image")
        prompt = kwargs.get("prompt")

        # The following can be isolated into a collate function
        if not isinstance(images, list):
            images = [images]
        if not isinstance(prompt, list):
            prompt = [prompt]
        for i, image in enumerate(images):
            if not isinstance(image, list):
                images[i] = [image]
        
        text_inputs = []
        for i, p in enumerate(prompt):
            message = [
                {
                    "role": "user", 
                    "content": [
                        *[{"type": "image"} for _ in range(len(images[i]))],
                        {"type": "text", "text": p}
                    ]
                }
            ]
            text_input = self.processor.apply_chat_template(
                message,
                add_generation_prompt=True,
            ).strip()
            text_inputs.append(text_input)
        
        # The actual processing of the inputs
        inputs = self.processor(
            text=text_inputs,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side="left",
        ).to(self.model.device, dtype=torch.bfloat16)
        return inputs

    def generate(self, inputs, **kwargs):
        return self.model.generate(
            **inputs,
            **kwargs,
        )

    def decode(self, generated_ids):
        decoded = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        return decoded
