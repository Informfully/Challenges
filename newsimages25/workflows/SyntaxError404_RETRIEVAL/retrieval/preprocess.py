from PIL import Image
from torchvision import transforms

# --- Image preprocessing ---
def preprocess_image(image, size=224):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])
    return transform(image).unsqueeze(0)  # add batch dim

# --- Text preprocessing ---
def preprocess_text(text, tokenizer):
    return tokenizer(text, return_tensors="pt", truncation=True, padding=True)
