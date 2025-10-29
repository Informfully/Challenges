from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from src.models.vlm_wrapper import VLMWrapperGeneration


@dataclass
class Qwen2VLWrapper(VLMWrapperGeneration):
    model: Any = field(
        default_factory=lambda: Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            device_map="auto"
        )
    )
    processor: Any = field(
        default_factory=lambda: AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    )

    def process_inputs(
            self,
            prompt: Union[str, List[str]],
            images: Union[Image.Image, List[Image.Image]],
            **kwargs,
    ) -> Dict[str, Any]:
        """
        Prepares model inputs from conversation and images.
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(images, Image.Image):
            images = [images]
        conversations = [[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {
                        "type": "text",
                        "text": f"{prompt[i % len(prompt)]}",
                    },
                ]
            }] for i in range(len(prompt) * len(images))
        ]
        text_prompt = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=kwargs.get("add_generation_prompt", True)
        )
        images_list = []
        for image in images:
            images_list.extend([image for _ in range(len(prompt))])
        inputs = self.processor(
            text=text_prompt,
            images=images_list,
            padding=kwargs.get("padding", True),
            return_tensors="pt"
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = inputs.to(device)
        return inputs

    def decode(
            self,
            generated_ids: Any,
            input_ids=None,
            return_generated_text_only: bool = False,
            **kwargs,
    ) -> str:
        if return_generated_text_only:
            assert input_ids is not None, "input_ids is required when return_generated_text_only is True"
        return self.processor.batch_decode(
            generated_ids if not return_generated_text_only else generated_ids[:, input_ids.shape[1]:],
            skip_special_tokens=kwargs.get('skip_special_tokens', True),
            clean_up_tokenization_spaces=kwargs.get('clean_up_tokenization_spaces', True)
        )

    def generate(self, inputs: Dict[str, Any], **kwargs) -> Any:
        output_ids = self.model.generate(**inputs, max_new_tokens=kwargs.get('max_new_tokens', 1024))
        return output_ids
