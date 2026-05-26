import torch
import diffusers

try:
    from sdnq import apply_sdnq_options_to_model
    triton_is_available = True
except ImportError:
    triton_is_available = False

ASPECT_RATIOS = {
    # "1:1": (1328, 1328),
    "1:1": (1024, 1024),
    # "16:9": (1664, 928),
    "16:9": (832, 464),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

POSITIVE_MAGIC = {
    "en": ", Ultra HD, 4K, cinematic composition.",
}

class QwenImageGenerator:
    """Wrapper around QwenImagePipeline with optional SDNQ INT8 quantization."""
    MODEL_ID = "Disty0/Qwen-Image-SDNQ-uint4-svd-r32"
    def __init__(
            self,
            torch_dtype: torch.dtype = torch.bfloat16,
            use_quantized_matmul: bool = True,
            cpu_offload: bool = True,
    ):
        self.torch_dtype = torch_dtype
        self.use_quantized_matmul = use_quantized_matmul
        self.cpu_offload = cpu_offload
        self.pipe = None

    def load(self) -> None:
        """Load the pipeline and apply SDNQ quantization if available."""
        print(f"Loading {self.MODEL_ID} …")
        self.pipe = diffusers.QwenImagePipeline.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.torch_dtype,
        )
        # Apply INT8 MatMul quantization for AMD, Intel ARC, and Nvidia GPUs
        if (
                self.use_quantized_matmul
                and triton_is_available
                and (torch.cuda.is_available() or torch.xpu.is_available())
        ):
            print("Applying SDNQ INT8 quantization …")
            self.pipe.transformer = apply_sdnq_options_to_model(
                self.pipe.transformer, use_quantized_matmul=True
            )
            self.pipe.text_encoder = apply_sdnq_options_to_model(
                self.pipe.text_encoder, use_quantized_matmul=True
            )
            # Uncomment for faster speeds (requires torch.compile support):
            # self.pipe.transformer = torch.compile(self.pipe.transformer)
        if self.cpu_offload:
            self.pipe.enable_model_cpu_offload()
        print("Model loaded.\n")

    def generate(
            self,
            prompt: str,
            negative_prompt: str = " ",
            aspect_ratio: str = "16:9",
            language: str = "en",
            num_inference_steps: int = 50,
            true_cfg_scale: float = 4.0,
            seed: int = 42,
            append_magic: bool = True,
    ):
        """
        Generate an image and return a PIL Image.

        Args:
            prompt:              The main text prompt.
            negative_prompt:     Concepts to avoid in the output.
            aspect_ratio:        Key from ASPECT_RATIOS (e.g. "16:9", "1:1").
            language:            Key from POSITIVE_MAGIC for the quality suffix ("en").
            num_inference_steps: Number of diffusion steps.
            true_cfg_scale:      Classifier-free guidance scale.
            seed:                RNG seed for reproducibility.
            append_magic:        Whether to append the positive magic suffix to the prompt.

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
        if language not in POSITIVE_MAGIC:
            raise ValueError(
                f"Unknown language '{language}'. "
                f"Choose from: {list(POSITIVE_MAGIC.keys())}"
            )
        width, height = ASPECT_RATIOS[aspect_ratio]
        full_prompt = prompt + POSITIVE_MAGIC[language] if append_magic else prompt
        result = self.pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        )
        return result.images[0]