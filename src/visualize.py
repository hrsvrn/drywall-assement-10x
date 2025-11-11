import cv2, matplotlib.pyplot as plt
import torch
from PIL import Image
import wandb
import numpy as np

def visualize(image_path, prompt, model, processor, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        pred = torch.sigmoid(model(**inputs).logits).squeeze().cpu().numpy()

    mask = (pred > 0.5).astype(np.uint8) * 255
    heatmap = cv2.applyColorMap((pred*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(image), 0.7, heatmap, 0.3, 0)

    plt.imshow(overlay)
    plt.title(prompt)
    plt.axis("off")
    plt.show()

    wandb.log({"Prediction": [wandb.Image(overlay, caption=prompt)]})
