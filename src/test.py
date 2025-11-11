import os
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# Paths
MODEL_PATH = "../checkpoints/clipseg_final.pth"
RESULTS_DIR = "../results/"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load processor + model
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("✅ Model loaded from:", MODEL_PATH)

# Prompts for each task
PROMPTS = ["segment crack", "segment taping area"]

def predict_and_save(image_path, prompt):
    """Run segmentation and save overlay + mask"""
    image = Image.open(image_path).convert("RGB")

    # Preprocess and forward pass
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()

    # Binary mask
    mask = (pred > 0.5).astype(np.uint8) * 255

    # Visualization overlay
    heatmap = cv2.applyColorMap((pred * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(image), 0.7, heatmap, 0.3, 0)

    # Save both
    base = os.path.basename(image_path).split('.')[0]
    cv2.imwrite(os.path.join(RESULTS_DIR, f"{base}_{prompt.replace(' ', '_')}_mask.png"), mask)
    cv2.imwrite(os.path.join(RESULTS_DIR, f"{base}_{prompt.replace(' ', '_')}_overlay.jpg"), overlay)

    # Optional: Display inline
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(overlay)
    plt.title(prompt)
    plt.axis("off")
    plt.show()

# ---------- Run Testing ----------

TEST_IMAGES = [
    "../processed_datasets/cracks-1/val/images/00005_jpg.rf.4d683f6e2fb6b93175f842b5bd75c7a8.jpg",
    "../processed_datasets/drywall-1/val/images/dw_034.jpg"
]

for img_path in TEST_IMAGES:
    for prompt in PROMPTS:
        predict_and_save(img_path, prompt)

print("✅ Inference complete! Results saved in:", RESULTS_DIR)
