import torch
from transformers import CLIPProcessor, CLIPModel

class CLIPEncoder:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def encode_text(self, text_list):
        inputs = self.processor(text=text_list, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb.cpu().numpy()

    def encode_image(self, image_list):
        inputs = self.processor(images=image_list, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb.cpu().numpy()
