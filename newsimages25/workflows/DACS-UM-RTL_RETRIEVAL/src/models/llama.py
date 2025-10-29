from typing import Any, Dict, List

from src.models.llm_wrapper import LLMWrapper


class Llama38BWrapper(LLMWrapper):
    def get_prompt(
            self,   
            messages: List[Dict[str, Any]],
            tokenize: bool = False,
            add_generation_prompt: bool = True,
            **kwargs) -> List[Dict[str, Any]]:
        """
        Apply the chat template to the messages.

        Args:
            messages: List of message dicts to apply the chat template to.
            tokenize: whether to tokenize the messages (if true will return input_ids and attention_mask)
            add_generation_prompt: whether to add the generation prompt to the messages.

        Message format:
        [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ]
        """
        for message in messages:
            assert message["role"] in ["user", "assistant", "system"]
            assert "content" in message
        return self.pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs
        )

    def generate(self, prompt: str, generation_config: Dict[str, Any]) -> str:
        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        model_output = self.pipeline(
            prompt,
            eos_token_id=terminators,
            pad_token_id=self.pipeline.model.config.eos_token_id,
            return_full_text=False,
            **generation_config
        )
        answer = model_output[0]["generated_text"]
        return answer