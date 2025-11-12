from torch.utils.data import Dataset
from PIL import Image
import torch
import pandas as pd
import numpy as np
import os
import albumentations as A


def collate_fn(batch):
    """
    Custom collate function for GroundedSAM.
    Returns batched images, prompts, and masks as numpy arrays.
    """
    images = [item["image"] for item in batch]
    prompts = [item["prompt"] for item in batch]
    masks = torch.stack([item["mask"] for item in batch])
    
    return {
        "images": images,  # List of numpy arrays
        "prompts": prompts,  # List of strings
        "masks": masks  # Tensor (B, H, W)
    }


class GroundedSAMDataset(Dataset):
    """
    Dataset for GroundingDINO + SAM text-conditioned segmentation.
    Returns images as numpy arrays, text prompts, and ground truth masks.
    """
    
    def __init__(self, csv_path, split="train", transform=None):
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data["split"] == split].reset_index(drop=True)
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
        
        # Load image and mask as numpy arrays
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L")) / 255.0

        # Apply augmentations
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]
        
        # Convert mask to tensor
        mask = torch.tensor(mask, dtype=torch.float32)
        
        return {
            "image": image,  # numpy array (H, W, 3)
            "prompt": row["prompt"],  # string
            "mask": mask  # tensor (H, W)
        }


# Backward compatibility alias
CLIPSegDataset = GroundedSAMDataset
