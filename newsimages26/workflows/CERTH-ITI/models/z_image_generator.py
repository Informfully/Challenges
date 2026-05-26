import torch
from diffusers import ZImagePipeline


ASPECT_RATIOS = {
    "1:1": (512, 512),
    # "16:9": (1664, 928),
    "16:9": (832, 464),

    "4:3":   (1024, 768),
    "9:16":  (928, 1664),
}


class ZImageGenerator:
    """Wrapper around ZImagePipeline for the Z-Image-Turbo model."""

    MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.dtype = dtype
        self.pipe = None

    def load(self) -> None:
        """Load the pipeline onto the target device."""
        self.pipe = ZImagePipeline.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=False,
        )
        self.pipe.to(self.device)

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        aspect_ratio: str = "16:9",
        num_inference_steps: int = 9,
        guidance_scale: float = 0.0,
        seed: int = 42,
    ):
        """
        Generate an image and return a PIL Image.

        Args:
            prompt:              Text prompt describing the desired image.
            negative_prompt:     Things to avoid in the generated image.
            aspect_ratio:        Key from ASPECT_RATIOS dict (e.g. "1:1", "16:9").
            num_inference_steps: Diffusion steps (9 → 8 DiT forwards for Turbo).
            guidance_scale:      Should be 0.0 for Turbo models.
            seed:                RNG seed for reproducibility.

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
        generator = torch.Generator(self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return result.images[0]
