import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


class Qwen3VLAssistant:
    """Wrapper around Qwen3-VL-8B-Instruct for text-based Q&A."""

    MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

    def __init__(
        self,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        # attn_implementation: str = "flash_attention_2",  # uncomment if flash-attn is installed
    ):
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.model = None
        self.processor = None

    def load(self) -> None:
        """Load the model and processor. First run downloads ~16 GB of weights."""
        print(f"Loading {self.MODEL_ID} … (first run downloads the weights, ~16 GB)")
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            # attn_implementation=self.attn_implementation,
        )
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        print("Model loaded.\n")

    def ask_about_text(
        self,
        context: str,
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> str:
        """
        Ask a question about a block of text.

        Args:
            context:        The source text the model should reason over.
            question:       The question to answer based on the context.
            max_new_tokens: Maximum tokens to generate in the answer.
            temperature:    Sampling temperature — lower is more factual.
            top_p:          Nucleus sampling probability threshold.

        Returns:
            The model's answer as a stripped string.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        messages = [
            {
                "role": "user",
                "content": f"Here is a text:\n{context}\nNow\n{question}",
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )

        # Strip the prompt tokens, keep only the newly generated part
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        answer = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
        )[0]

        return answer.strip()

    def ask_for_improved_prompt(
            self,
            history_str: str,
            article_title: str,
            max_new_tokens: int = 512,
            temperature: float = 0.7,
            top_p: float = 0.9,
    ) -> str:
        """
        Ask the model to generate an improved image prompt based on past attempts.

        Args:
            history_str:    A formatted string of previous (prompt, score) attempts.
            max_new_tokens: Maximum tokens to generate. 256 is generous for a single prompt.
            temperature:    Sampling temperature — higher allows more creative suggestions.
            top_p:          Nucleus sampling probability threshold.

        Returns:
            The newly generated prompt as a stripped string.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        meta_prompt = (
            f"You are a prompt optimization expert.\n"
            f"Your goal is to generate an image generation prompt that best captures the news article with title {article_title}.\n\n"
            f"Here is the history of previous attempts and their scores (1–5, higher is better):\n"
            f"{history_str}\n\n"
            f"Analyze what made higher-scoring prompts better and lower-scoring ones worse.\n"
            f"Generate a new prompt that improves on these attempts.\n"
            f"Output only the prompt, with no explanation or additional text."
        )

        messages = [
            {
                "role": "user",
                "content": meta_prompt,
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
        )[0]
        return answer.strip()