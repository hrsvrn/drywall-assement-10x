from torch.utils.data import Dataset
from PIL import Image
import torch
import pandas as pd
import numpy as np
import os
import albumentations as A

class CLIPSegDataset(Dataset):
    def __init__(self, csv_path, processor, split="train", transform=None):
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data["split"] == split].reset_index(drop=True)
        self.processor = processor
        self.transform = transform
        
        # Get root directory (parent of src)
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Resolve paths relative to root directory
        image_path = os.path.join(self.root_dir, row["image_path"])
        mask_path = os.path.join(self.root_dir, row["mask_path"])
        
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L")) / 255.0
        mask = torch.tensor(mask).unsqueeze(0).float()

        if self.transform:
            aug = self.transform(image=image, mask=mask.squeeze().numpy())
            image, mask = aug["image"], torch.tensor(aug["mask"]).unsqueeze(0)

        image = Image.fromarray(image)
        inputs = self.processor(text=row["prompt"], images=image, return_tensors="pt")
        inputs["mask"] = mask
        return inputs
