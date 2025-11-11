import torch
from torch.utils.data import DataLoader
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from tqdm import tqdm
import wandb
from dataset import CLIPSegDataset
from loss import hybrid_loss
from evaluate import evaluate
from utils import get_train_transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

wandb.init(
    project="prompted-segmentation-drywall-qa",
    config={
        "model": "CLIPSeg (CIDAS/clipseg-rd64-refined)",
        "epochs": 15,
        "batch_size": 4,
        "lr": 5e-5,
        "loss": "BCE + Dice + Edge-aware",
        "augmentations": "CLAHE + Brightness + Flip + Blur",
    }
)

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

for param in model.text_encoder.parameters():
    param.requires_grad = False

train_ds = CLIPSegDataset("../processed_datasets/dataset.csv", processor, split="train", transform=get_train_transforms())
val_ds   = CLIPSegDataset("../processed_datasets/dataset.csv", processor, split="valid")

train_dl = DataLoader(train_ds, batch_size=4, shuffle=True)
val_dl   = DataLoader(val_ds, batch_size=2)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(15):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}"):
        masks = batch.pop("mask").to(device)
        inputs = {k:v.squeeze().to(device) for k,v in batch.items()}
        outputs = model(**inputs)
        loss = hybrid_loss(outputs.logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        wandb.log({"train_loss": loss.item()})

    avg_loss = total_loss / len(train_dl)
    wandb.log({"epoch_loss": avg_loss})
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")

    val_dice, val_iou = evaluate(model, val_dl, device)
    wandb.log({"val_dice": val_dice, "val_mIoU": val_iou})
    print(f"Validation | Dice: {val_dice:.3f} | mIoU: {val_iou:.3f}")

torch.save(model.state_dict(), "../checkpoints/clipseg_final.pth")
wandb.save("../checkpoints/clipseg_final.pth")
