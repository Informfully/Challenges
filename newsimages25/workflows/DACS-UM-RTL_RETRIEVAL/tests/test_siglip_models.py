import numpy as np
import pytest
from PIL import Image
from transformers import AutoProcessor, SiglipModel, SiglipProcessor

from src.models.configs import get_model_config
from src.models.siglip import SigLipWrapper

BATCH_SIZE = 10

# model_id, embedding_dim
SIGLIP_MODEL_CONFIGS = [
    ("google/siglip-base-patch16-256", 768),
    ("google/siglip-large-patch16-256", 1024),
]


def _get_siglip_model_config(model_id):
    return get_model_config("siglip", model_id)


def _get_siglip_model_processor(model_id):
    config = _get_siglip_model_config(model_id)
    return (
        config["processor_class"].from_pretrained(config["model_id"]),
        config["model_class"].from_pretrained(config["model_id"]),
    )


def _get_random_image_batch(batch_size: int):
    random_array = np.random.randint(0, 255, (batch_size, 100, 100, 3), dtype=np.uint8)
    temp_image = [Image.fromarray(random_array[i]) for i in range(batch_size)]
    return temp_image


def _get_random_text_batch(batch_size: int):
    prompts = [
        "A photo of a cat",
        "A photo of a dog",
        "A photo of a bird",
        "A photo of a fish",
        "A photo of a horse",
    ]
    return [prompts[i % len(prompts)] for i in range(batch_size)]


@pytest.mark.parametrize("model_id, embedding_dim", SIGLIP_MODEL_CONFIGS)
def test_siglip__init_config_default(model_id, embedding_dim):
    config = _get_siglip_model_config(model_id)
    assert config["model_id"] == model_id, (
        f"Model ID mismatch: {config['model_id']} != {model_id}"
    )
    assert config["model_class"] == SiglipModel, (
        f"Model class mismatch: {config['model_class']} != SiglipModel"
    )
    assert config["processor_class"] == AutoProcessor, (
        f"Processor class mismatch: {config['processor_class']} != AutoProcessor"
    )
    assert config["wrapper_class"] == SigLipWrapper, (
        f"Wrapper class mismatch: {config['wrapper_class']} != SigLipWrapper"
    )


@pytest.mark.parametrize("model_id, embedding_dim", SIGLIP_MODEL_CONFIGS)
def test_siglip__init_model_processor_default(model_id, embedding_dim):
    processor, model = _get_siglip_model_processor(model_id)
    assert isinstance(processor, SiglipProcessor), (
        f"Processor is not a SiglipProcessor: {model_id}"
    )
    assert isinstance(model, SiglipModel), f"Model is not a SiglipModel: {model_id}"


@pytest.mark.parametrize("model_id, embedding_dim", SIGLIP_MODEL_CONFIGS)
def test_siglip__get_image_features(model_id, embedding_dim):
    processor, model = _get_siglip_model_processor(model_id)
    wrapper = SigLipWrapper(model=model, processor=processor)
    images = _get_random_image_batch(BATCH_SIZE)
    inputs = wrapper.process_inputs(images=images)
    image_features = wrapper.get_image_embeddings(inputs)
    assert image_features.shape == (BATCH_SIZE, embedding_dim), (
        f"{model_id} image features shape mismatch"
    )


@pytest.mark.parametrize("model_id, embedding_dim", SIGLIP_MODEL_CONFIGS)
def test_siglip__get_text_features(model_id, embedding_dim):
    processor, model = _get_siglip_model_processor(model_id)
    wrapper = SigLipWrapper(model=model, processor=processor)
    text = _get_random_text_batch(BATCH_SIZE)
    inputs = wrapper.process_inputs(text=text)
    text_features = wrapper.get_text_embeddings(inputs)
    assert text_features.shape == (BATCH_SIZE, embedding_dim), (
        f"{model_id} text features shape mismatch"
    )

@pytest.mark.parametrize("model_id, embedding_dim", SIGLIP_MODEL_CONFIGS)
def test_siglip__get_embeddings(model_id, embedding_dim):
    processor, model = _get_siglip_model_processor(model_id)
    wrapper = SigLipWrapper(model=model, processor=processor)
    images = _get_random_image_batch(BATCH_SIZE)
    text = _get_random_text_batch(BATCH_SIZE)
    inputs = wrapper.process_inputs(images=images, text=text)
    embeddings = wrapper.get_embeddings(inputs)
    assert embeddings['image_embeds'].shape == (BATCH_SIZE, embedding_dim), (
        f"{model_id} image embeddings shape mismatch"
    )
    assert embeddings['text_embeds'].shape == (BATCH_SIZE, embedding_dim), (
        f"{model_id} text embeddings shape mismatch"
    )
    assert embeddings['logits_per_image'].shape == (BATCH_SIZE, BATCH_SIZE), (
        f"{model_id} logits_per_image shape mismatch"
    )
    assert embeddings['logits_per_text'].shape == (BATCH_SIZE, BATCH_SIZE), (
        f"{model_id} logits_per_text shape mismatch"
    )
