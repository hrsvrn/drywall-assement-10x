import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from dataset import CLIPSegDataset
from evaluate import evaluate
from visualize import visualize

CSV_PATH = "../processed_datasets/dataset.csv"
MODEL_PATH = "../checkpoints/clipseg_final.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Model checkpoint loaded:", MODEL_PATH)

# Load validation data
val_ds = CLIPSegDataset(CSV_PATH, processor, split="valid")
val_dl = DataLoader(val_ds, batch_size=2, shuffle=False)

# Evaluate model
print("Running evaluation on validation set...")
dice_score, miou_score = evaluate(model, val_dl, device)
print(f"Validation Dice: {dice_score:.4f} | mIoU: {miou_score:.4f}")

# Visualize a few predictions
print("Displaying a few validation samples...")
for i in range(2):
    row = val_ds.data.iloc[i]
    visualize(row["image_path"], row["prompt"], model, processor, device)
