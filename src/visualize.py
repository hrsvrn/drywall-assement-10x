import cv2, matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
from PIL import Image
import wandb
import numpy as np

def visualize(image_path, prompt, model, processor, device):
    # Get root directory if path is relative
    if not os.path.isabs(image_path):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        image_path = os.path.join(root_dir, image_path)
    
    image = Image.open(image_path).convert("RGB")
    orig_size = image.size  # (width, height)
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        pred = torch.sigmoid(model(**inputs).logits)  # Keep as tensor
        
        # Resize prediction to original image size
        pred = F.interpolate(
            pred.unsqueeze(0),  # Add batch dim
            size=(orig_size[1], orig_size[0]),  # (height, width)
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()

    mask = (pred > 0.5).astype(np.uint8) * 255
    heatmap = cv2.applyColorMap((pred*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(image), 0.7, heatmap, 0.3, 0)

    plt.imshow(overlay)
    plt.title(prompt)
    plt.axis("off")
    plt.show()

    wandb.log({"Prediction": [wandb.Image(overlay, caption=prompt)]})
