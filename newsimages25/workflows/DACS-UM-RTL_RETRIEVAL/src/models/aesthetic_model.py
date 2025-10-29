import os
from os.path import expanduser  # pylint: disable=import-outside-toplevel
from typing import Any, List, Optional
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel

import open_clip
import torch
import torch.nn as nn

# Code adapted from repository: https://github.com/LAION-AI/aesthetic-predictor/


class AestheticModel(nn.Module):

    def __init__(
            self,
            model_name: Optional[str] = None,
            local_path: Optional[str] = None,
            device: str = 'cpu',
    ):
        super().__init__()
        checkpoint_model_name = model_name.replace("-", "_").lower()
        self.forward_model = AestheticModel.get_aesthetic_model(checkpoint_model_name, local_path)
        self.clip_model, _, self.preprocess =\
            open_clip.create_model_and_transforms(model_name, pretrained='openai')
        self.clip_model.eval()
        self.device = device 
        self.clip_model.to(device)
        self.forward_model.to(device)

    def forward(self, images: List[Any]):
        images = [self.preprocess(image).to(self.device) for image in images]
        with torch.no_grad():
            image_features = self.clip_model.encode_image(torch.stack(images))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            prediction = self.forward_model(image_features).detach().squeeze().to('cpu')
        return prediction
            
    @staticmethod
    def get_aesthetic_model(clip_model="vit_l_14", local_path: Optional[str] = None):
        """load the aethetic model"""
        assert clip_model in ["vit_l_14", "vit_b_32"], f"Clip model {clip_model} not supported"
        if local_path is not None:
            assert os.path.exists(local_path), f"Local path {local_path} does not exist"
            assert clip_model in local_path
            path_to_model = local_path
        else:
            home = expanduser("~")
            cache_folder = home + "/.cache/emb_reader"
            path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
            if not os.path.exists(path_to_model):
                os.makedirs(cache_folder, exist_ok=True)
                url_model = (
                    "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth"
                )
                urlretrieve(url_model, path_to_model)
        if clip_model == "vit_l_14":
            m = nn.Linear(768, 1)
        elif clip_model == "vit_b_32":
            m = nn.Linear(512, 1)
        s = torch.load(path_to_model)
        m.load_state_dict(s)
        m.eval()
        return m