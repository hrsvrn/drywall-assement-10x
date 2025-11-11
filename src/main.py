import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from tqdm import tqdm
import wandb
from dataset import CLIPSegDataset, collate_fn
from loss import hybrid_loss
from evaluate import evaluate
from visualize import visualize
from utils import get_train_transforms

# --------------------------
# CONFIGURATION
# --------------------------
CSV_PATH = "../processed_datasets/dataset.csv"
MODEL_SAVE_PATH = "../checkpoints/clipseg_final.pth"
RESULTS_DIR = "../results/"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("../checkpoints", exist_ok=True)

EPOCHS = 15
BATCH_SIZE = 4
LR = 5e-5

# --------------------------
# INITIAL SETUP
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Initialize W&B
wandb.init(
    project="prompted-segmentation-drywall-qa",
    config={
        "model": "CLIPSeg (CIDAS/clipseg-rd64-refined)",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "loss": "BCE + Dice + Edge",
        "dataset": "Cracks + Drywall",
        "augmentations": "CLAHE, brightness, flip, blur"
    }
)

# --------------------------
# LOAD MODEL AND DATA
# --------------------------
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

# Freeze text encoder (CLIP text model)
for param in model.clip.text_model.parameters():
    param.requires_grad = False

# Dataset and Dataloaders
train_ds = CLIPSegDataset(CSV_PATH, processor, split="train", transform=get_train_transforms())
val_ds   = CLIPSegDataset(CSV_PATH, processor, split="valid")

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dl   = DataLoader(val_ds, batch_size=2, collate_fn=collate_fn)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# --------------------------
# TRAINING LOOP
# --------------------------
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        masks = batch.pop("mask").to(device)
        inputs = {k:v.to(device) for k,v in batch.items()}

        outputs = model(**inputs)
        loss = hybrid_loss(outputs.logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        wandb.log({"train_loss": loss.item()})

    avg_train_loss = total_loss / len(train_dl)
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_dice, val_miou = evaluate(model, val_dl, device)
    wandb.log({
        "epoch": epoch+1,
        "epoch_train_loss": avg_train_loss,
        "val_dice": val_dice,
        "val_mIoU": val_miou
    })
    print(f"Validation Dice: {val_dice:.4f} | mIoU: {val_miou:.4f}")

# --------------------------
# SAVE MODEL
# --------------------------
torch.save(model.state_dict(), MODEL_SAVE_PATH)
wandb.save(MODEL_SAVE_PATH)
print("Model saved to:", MODEL_SAVE_PATH)

# --------------------------
# VALIDATION VISUALIZATION
# --------------------------
print("Generating a few validation samples...")
for i in range(2):
    row = val_ds.data.iloc[i]
    visualize(row["image_path"], row["prompt"], model, processor, device)

print("Pipeline complete. All results and metrics logged to W&B.")
