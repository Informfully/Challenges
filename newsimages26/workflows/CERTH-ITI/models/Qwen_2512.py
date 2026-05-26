import torch
from diffusers import (
    QwenImagePipeline,
    QwenImageTransformer2DModel,
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
)
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig as TransformersBitsAndBytesConfig,
)


ASPECT_RATIOS = {
    "1:1":  (1024, 1024),
    # "16:9": (1664, 928),
    "16:9": (832, 464),
    "9:16": (928, 1664),
    "4:3":  (1472, 1140),
    "3:4":  (1140, 1472),
    "3:2":  (1584, 1056),
    "2:3":  (1056, 1584),
}


class QwenImage2512Generator:
    """
    Wrapper around QwenImagePipeline for Qwen-Image-2512.
    Automatically applies 4-bit NF4 quantization on CUDA,
    and falls back to float32 on CPU.
    """

    MODEL_ID = "Qwen/Qwen-Image-2512"

    def __init__(self, cpu_offload: bool = True):
        self.cpu_offload = cpu_offload
        self.pipe = None

        if torch.cuda.is_available():
            self.device = "cuda"
            self.torch_dtype = torch.bfloat16
        else:
            self.device = "cpu"
            self.torch_dtype = torch.float32

    def _build_quant_configs(self):
        """Return (transformer_quant_config, text_encoder_quant_config)."""
        if self.device != "cuda":
            return None, None

        transformer_quant_config = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["transformer_blocks.0.img_mod"],  # avoids artifacts
        )
        text_encoder_quant_config = TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        return transformer_quant_config, text_encoder_quant_config

    def load(self) -> None:
        """Load transformer, text encoder, and assemble the pipeline."""
        print(f"Loading {self.MODEL_ID} on {self.device} …")

        transformer_quant_config, text_encoder_quant_config = self._build_quant_configs()

        print("  Loading transformer …")
        transformer = QwenImageTransformer2DModel.from_pretrained(
            self.MODEL_ID,
            subfolder="transformer",
            quantization_config=transformer_quant_config,
            device_map="cpu",
            torch_dtype=self.torch_dtype,
        )

        print("  Loading text encoder …")
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            subfolder="text_encoder",
            quantization_config=text_encoder_quant_config,
            device_map="cpu",
            torch_dtype=self.torch_dtype,
        )

        print("  Assembling pipeline …")
        self.pipe = QwenImagePipeline.from_pretrained(
            self.MODEL_ID,
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=self.torch_dtype,
        )

        if self.cpu_offload:
            self.pipe.enable_model_cpu_offload()

        print("Model loaded.\n")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = " ",
        aspect_ratio: str = "16:9",
        num_inference_steps: int = 50,
        seed: int | None = None,
    ):
        """
        Generate an image and return a PIL Image.

        Args:
            prompt:              Text prompt describing the desired image.
            aspect_ratio:        Key from ASPECT_RATIOS (e.g. "16:9", "1:1").
            num_inference_steps: Number of diffusion steps.
            seed:                RNG seed for reproducibility. None for random.

        Returns:
            PIL.Image.Image
        """
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded. Call .load() first.")

        if aspect_ratio not in ASPECT_RATIOS:
            raise ValueError(
                f"Unknown aspect ratio '{aspect_ratio}'. "
                f"Choose from: {list(ASPECT_RATIOS.keys())}"
            )

        width, height = ASPECT_RATIOS[aspect_ratio]

        generator = (
            torch.Generator(device=self.device).manual_seed(seed)
            if seed is not None
            else None
        )

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        return result.images[0]