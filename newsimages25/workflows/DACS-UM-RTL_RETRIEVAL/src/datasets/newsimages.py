import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.models.vlm_wrapper import VLMWrapperRetrieval


class NewsImagesDataHandler:
    """
    A class to handle the data for the NewsImages dataset:
    - split the data into train, val, and test sets.
    If subset flag = True, it returns a specified subset based on indices provided
    """
    def __init__(self, csv_path: str, image_path: Optional[str] = None):
        self.csv_path = csv_path
        self.image_path = image_path
    
    def get_subset(self, subset: str):
        subset_df = pd.read_csv(subset, header=None, index_col=None)
        subset_df.columns = ["article_id", "article_url", "article_title", "article_tags", "image_id", "image_url"]
        subset_df["article_id_num"] = subset_df["article_id"]
        return subset_df
    
    def split_csv_train_val_test(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: Optional[int] = None,
    ):  
        assert abs(train_ratio + val_ratio + test_ratio - 1) < 1e-9

        if seed is not None:
            np.random.seed(seed)

        data_csv = pd.read_csv(self.csv_path)
        data_csv["article_id_num"] = list(range(len(data_csv)))
        # shuffle the data for splits
        data_csv = data_csv.sample(frac=1).reset_index(drop=True)
        train_size = int(len(data_csv) * train_ratio)
        val_size = int(len(data_csv) * val_ratio)

        train_data = data_csv.iloc[:train_size]
        val_data = data_csv.iloc[train_size:train_size + val_size]
        test_data = data_csv.iloc[train_size + val_size:]

        return train_data, val_data, test_data


class NewsImageDataset(Dataset):
    def __init__(
        self,
        image_path: Optional[str] = None,
        csv_path: Optional[str] = None,
        dataset_df: Optional[pd.DataFrame] = None,
        get_images: bool = True,
        use_article_id_num: bool = True,
    ):  
        self.image_path = image_path
        self.get_images = get_images
        self.use_article_id_num = use_article_id_num

        if self.get_images and self.image_path is None:
            raise ValueError("image_path must be provided if get_images is True")

        if csv_path is not None:
            self.data = self._prepare_data(image_path, csv_path=csv_path)
        elif dataset_df is not None:
            self.data = self._prepare_data(image_path, dataset_df=dataset_df)
        else:
            raise ValueError("Either csv_path or dataset_df must be provided")
    
    def _prepare_data(
        self,
        image_path: Optional[str] = None,
        dataset_df: Optional[pd.DataFrame] = None,
        csv_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Prepare the data for querying with index: map image paths to rows in the dataset.
        Args:
            dataset_df: the dataset dataframe (from NewsImagesManager.split_csv_train_val_test)
            image_path: the path to the images
        """
        if csv_path is not None:
            dataset_df = pd.read_csv(csv_path)
        elif dataset_df is not None:
            pass
        else:
            raise ValueError("Either csv_path or dataset_df must be provided")

        if image_path is not None:
            image_paths = {
                os.path.splitext(img)[0]: os.path.join(image_path, img) \
                    for img in os.listdir(image_path)
            }
            dataset_df["image_path"] = dataset_df["image_id"].map(lambda x: image_paths[x])
        return dataset_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        item = {
            "article_id": row["article_id_num"] if self.use_article_id_num else row["article_id"],
            "title": row["article_title"],
            "url": row["article_url"],
            "tags": row["article_tags"].split(";"),
        }
        
        if self.image_path is not None:
            item["image_path"] = row["image_path"]

        if self.get_images:
            item["image"] = Image.open(row["image_path"]).convert("RGB")
        
        return item

class NewsImagesCollator:
    def __init__(
        self,
        wrapper: Optional[VLMWrapperRetrieval] = None,
        process_images: bool = True,
        process_titles: bool = True,
        process_captions: bool = False,
        num_captions: Optional[int] = None,
    ):
        self.wrapper = wrapper
        self.process_images = process_images
        self.process_titles = process_titles
        self.process_captions = process_captions
        self.num_captions = num_captions
        # Default number of captions is 5
        if self.process_captions and self.num_captions is None:
            self.num_captions = 5
    
    def __call__(self, batch):
        if self.process_images:
            images = [item["image"] for item in batch]
        else:
            images = None

        titles = [item["title"] for item in batch]

        processed_batch = {}

        # For now, we process titles with VLM processor
        # We can also consider adding tags either separately or as part of the title
        if self.process_images and self.process_titles:
            processed_batch = self.wrapper.process_inputs(images=images, text=titles)
        elif self.process_images:   
            processed_batch = self.wrapper.process_inputs(images=images)
        elif self.process_titles:
            processed_batch = self.wrapper.process_inputs(text=titles)
        
        if self.process_captions:
            assert batch[0].get("captions") is not None, "captions must be provided"
            processed_batch["captions"] = self._process_captions([item["captions"] for item in batch])
        
        if self.process_images:
            processed_batch["image_paths"] = [item["image_path"] for item in batch]
        else:
            processed_batch["image_paths"] = None

        processed_batch["titles"] = titles
        processed_batch["tags"] = [item["tags"] for item in batch]
        processed_batch["article_id"] = [item["article_id"] for item in batch]
        
        return processed_batch
    
    def _process_captions(self, captions: List[List[str]]) -> Dict[str, torch.Tensor]:
        """
        Process captions with the VLM processor's tokenizer. 
        It is possible that some captions have less or more than `self.num_captions` caption:
            pad or truncate to `self.num_captions`.

        Args:
            captions: list of lists of captions
        """
        captions_flattened = []
        for sublist in captions:
            if len(sublist) < self.num_captions:
                padding_length = self.num_captions - len(sublist)
                sublist = sublist + [""] * padding_length
            elif len(sublist) > self.num_captions:
                sublist = sublist[:self.num_captions]
            captions_flattened.extend(sublist)

        processed_captions = self.wrapper.process_inputs(text=captions_flattened)

        for key in processed_captions.keys():
            processed_captions[key] = processed_captions[key].view(len(captions), self.num_captions, -1)
        
        return processed_captions