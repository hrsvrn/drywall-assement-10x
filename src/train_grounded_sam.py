"""
Training script for GroundingDINO + SAM text-conditioned segmentation.

This script fine-tunes SAM's mask decoder while using GroundingDINO as a frozen detector.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from dataset import GroundedSAMDataset, collate_fn
from model import GroundedSAM
from loss import hybrid_loss, dice_loss, focal_loss
from evaluate import evaluate_grounded_sam
from utils import get_train_transforms
import numpy as np

# --------------------------
# CONFIGURATION
# --------------------------
CSV_PATH = "../processed_datasets/dataset.csv"
MODEL_SAVE_PATH = "../checkpoints/grounded_sam_final.pth"
RESULTS_DIR = "../results/"

# Model checkpoints (download these first!)
GROUNDING_DINO_CONFIG = "../checkpoints/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "../checkpoints/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT = "../checkpoints/sam_vit_h_4b8939.pth"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("../checkpoints", exist_ok=True)

# Training hyperparameters
EPOCHS = 20
BATCH_SIZE = 2  # Smaller batch size due to SAM's memory requirements
LR = 1e-5  # Lower learning rate for fine-tuning
SEED = 42

# --------------------------
# SETUP
# --------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize W&B
wandb.init(
    project="prompted-segmentation-drywall-qa",
    config={
        "model": "GroundingDINO + SAM",
        "grounding_dino": "SwinT-OGC",
        "sam": "ViT-H",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "seed": SEED,
        "loss": "Dice + Focal",
        "dataset": "Cracks + Drywall",
        "strategy": "Fine-tune SAM decoder, freeze GroundingDINO"
    }
)

# --------------------------
# LOAD MODEL
# --------------------------
print("Loading GroundingDINO + SAM model...")
model = GroundedSAM(
    grounding_dino_config_path=GROUNDING_DINO_CONFIG,
    grounding_dino_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    sam_checkpoint_path=SAM_CHECKPOINT,
    sam_encoder_version="vit_h",
    box_threshold=0.25,
    text_threshold=0.25,
    device=device
)

# Freeze GroundingDINO (we only fine-tune SAM)
if model.grounding_dino:
    for param in model.grounding_dino.model.parameters():
        param.requires_grad = False

# Freeze SAM image encoder, fine-tune prompt encoder and mask decoder
if model.sam_predictor:
    # Freeze image encoder (large, pre-trained)
    for param in model.sam_predictor.model.image_encoder.parameters():
        param.requires_grad = False
    
    # Enable prompt encoder (needed for box encoding with gradients)
    for param in model.sam_predictor.model.prompt_encoder.parameters():
        param.requires_grad = True
    
    # Enable mask decoder (main training target)
    for param in model.sam_predictor.model.mask_decoder.parameters():
        param.requires_grad = True

print("Model loaded successfully!")

# --------------------------
# LOAD DATA
# --------------------------
print("Loading datasets...")
train_ds = GroundedSAMDataset(CSV_PATH, split="train", transform=get_train_transforms())
val_ds = GroundedSAMDataset(CSV_PATH, split="valid")

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
val_dl = DataLoader(val_ds, batch_size=1, collate_fn=collate_fn, num_workers=2)

print(f"Train samples: {len(train_ds)}")
print(f"Val samples: {len(val_ds)}")

# --------------------------
# OPTIMIZER
# --------------------------
# Optimize SAM's prompt encoder and mask decoder parameters
trainable_params = list(model.sam_predictor.model.prompt_encoder.parameters()) + \
                  list(model.sam_predictor.model.mask_decoder.parameters())
trainable_params = [p for p in trainable_params if p.requires_grad]

print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# --------------------------
# TRAINING LOOP
# --------------------------
print("Starting training...")
best_dice = 0.0

for epoch in range(EPOCHS):
    model.sam_predictor.model.train()  # Set SAM to training mode
    total_loss = 0.0
    
    pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in pbar:
        images = batch["images"]  # List of numpy arrays
        prompts = batch["prompts"]  # List of strings
        gt_masks = batch["masks"].to(device)  # (B, H, W)
        
        batch_loss = 0.0
        
        # Process each sample in the batch
        for i, (image, prompt) in enumerate(zip(images, prompts)):
            try:
                # Extract text label for GroundingDINO
                clean_prompt = prompt.replace("segment ", "").strip()
                
                # Step 1: Detect with GroundingDINO (no gradients needed)
                with torch.no_grad():
                    detections = model.grounding_dino.predict_with_classes(
                        image=image,
                        classes=[clean_prompt],
                        box_threshold=model.box_threshold,
                        text_threshold=model.text_threshold
                    )
                
                # If no detections, skip this sample
                if len(detections.xyxy) == 0:
                    print(f"\nNo detections for prompt: {prompt}")
                    continue
                
                # Step 2: Prepare and encode image with SAM
                # Use SAM's built-in image preprocessing which handles resizing correctly
                with torch.no_grad():
                    # This properly resizes to 1024x1024 and generates 64x64 feature maps
                    model.sam_predictor.set_image(image)
                    image_embedding = model.sam_predictor.features  # Get the cached features
                
                # Step 3: Transform boxes to SAM's coordinate space
                boxes = detections.xyxy
                boxes_tensor = torch.tensor(boxes, device=device, dtype=torch.float)
                
                # Apply SAM's transform to boxes (accounts for resizing/padding)
                boxes_transformed = model.sam_predictor.transform.apply_boxes_torch(
                    boxes_tensor,
                    image.shape[:2]
                )
                
                # Reshape boxes for SAM: (N, 2, 2) format [top-left, bottom-right]
                boxes_xyxy = boxes_transformed.reshape(-1, 2, 2)
                
                # Encode box prompts through prompt encoder
                sparse_embeddings, dense_embeddings = model.sam_predictor.model.prompt_encoder(
                    points=None,
                    boxes=boxes_xyxy,
                    masks=None,
                )
                
                # Get mask predictions from decoder (this maintains gradients!)
                low_res_masks, iou_predictions = model.sam_predictor.model.mask_decoder(
                    image_embeddings=image_embedding.repeat(boxes_xyxy.shape[0], 1, 1, 1),
                    image_pe=model.sam_predictor.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                
                # Get original image size
                h, w = image.shape[:2]
                
                # Upsample masks directly to original image size
                # (SAM will handle the transformation internally)
                upscaled_masks = F.interpolate(
                    low_res_masks,
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                )
                
                # Apply sigmoid
                upscaled_masks = torch.sigmoid(upscaled_masks)
                
                # Combine masks (union) - maintains gradients
                pred_mask = upscaled_masks.max(dim=0)[0]  # (1, H, W) with gradients!
                
                # Get corresponding GT mask and ensure it's float
                gt_mask = gt_masks[i:i+1].float()  # (1, H, W) float in [0, 1]
                
                # Resize prediction to match GT if needed
                if pred_mask.shape != gt_mask.shape:
                    pred_mask = F.interpolate(
                        pred_mask.unsqueeze(0),
                        size=gt_mask.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                
                # Compute loss (pred_mask has gradients now!)
                loss = dice_loss(pred_mask, gt_mask) + focal_loss(pred_mask, gt_mask)
                batch_loss += loss
                
            except Exception as e:
                print(f"\nError processing sample {i}: {e}")
                continue
        
        # Check if we have any valid loss
        if batch_loss == 0:
            print("\nWarning: No valid samples in batch, skipping...")
            continue
        
        # Average loss over batch
        batch_loss = batch_loss / len(images)
        
        # Backward pass
        optimizer.zero_grad()
        batch_loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        
        optimizer.step()
        
        total_loss += batch_loss.item()
        pbar.set_postfix({"loss": f"{batch_loss.item():.4f}"})
        wandb.log({"train_loss": batch_loss.item()})
    
    avg_train_loss = total_loss / len(train_dl)
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")
    
    # --------------------------
    # VALIDATION
    # --------------------------
    model.sam_predictor.model.eval()  # Set SAM to eval mode
    val_dice, val_miou = evaluate_grounded_sam(model, val_dl, device)
    
    wandb.log({
        "epoch": epoch+1,
        "epoch_train_loss": avg_train_loss,
        "val_dice": val_dice,
        "val_mIoU": val_miou,
        "lr": optimizer.param_groups[0]['lr']
    })
    
    print(f"Validation - Dice: {val_dice:.4f} | mIoU: {val_miou:.4f}")
    
    # Save best model
    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.sam_predictor.model.state_dict(), MODEL_SAVE_PATH)
        wandb.save(MODEL_SAVE_PATH)
        print(f"✓ Best model saved! (Dice: {best_dice:.4f})")
    
    # Step scheduler
    scheduler.step()

# --------------------------
# FINAL SAVE
# --------------------------
final_path = MODEL_SAVE_PATH.replace("_final.pth", "_last.pth")
torch.save(model.sam_predictor.model.state_dict(), final_path)
wandb.save(final_path)
print(f"Training complete! Final model saved to: {final_path}")
print(f"Best validation Dice: {best_dice:.4f}")

wandb.finish()

