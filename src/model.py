"""
GroundingDINO + SAM Model Wrapper
Text-conditioned segmentation using GroundingDINO for detection and SAM for segmentation
"""

import torch
import numpy as np
from PIL import Image
import supervision as sv
from typing import List, Tuple
import os

try:
    from groundingdino.util.inference import Model as GroundingDINO
    GROUNDINGDINO_AVAILABLE = True
except ImportError:
    GROUNDINGDINO_AVAILABLE = False
    print("Warning: GroundingDINO not available. Install with: pip install groundingdino-py")

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: SAM not available. Install with: pip install segment-anything")


class GroundedSAM:
    """
    Combined GroundingDINO + SAM model for text-conditioned segmentation.
    
    Pipeline:
    1. GroundingDINO: text prompt → bounding boxes
    2. SAM: bounding boxes → segmentation masks
    """
    
    def __init__(
        self,
        grounding_dino_config_path: str = None,
        grounding_dino_checkpoint_path: str = None,
        sam_checkpoint_path: str = None,
        sam_encoder_version: str = "vit_h",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        device: str = "cuda"
    ):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        # Initialize GroundingDINO
        if GROUNDINGDINO_AVAILABLE and grounding_dino_config_path and grounding_dino_checkpoint_path:
            self.grounding_dino = GroundingDINO(
                model_config_path=grounding_dino_config_path,
                model_checkpoint_path=grounding_dino_checkpoint_path,
                device=device
            )
        else:
            self.grounding_dino = None
            print("Warning: GroundingDINO not initialized")
        
        # Initialize SAM
        if SAM_AVAILABLE and sam_checkpoint_path:
            sam = sam_model_registry[sam_encoder_version](checkpoint=sam_checkpoint_path)
            sam.to(device=device)
            self.sam_predictor = SamPredictor(sam)
        else:
            self.sam_predictor = None
            print("Warning: SAM not initialized")
    
    def predict(
        self, 
        image: np.ndarray, 
        text_prompt: str,
        return_boxes: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run text-conditioned segmentation.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            text_prompt: Natural language prompt (e.g., "crack", "drywall joint")
            return_boxes: Whether to also return bounding boxes
            
        Returns:
            masks: Binary segmentation mask (H, W) with values [0, 1]
            boxes: Bounding boxes (N, 4) if return_boxes=True
        """
        if self.grounding_dino is None or self.sam_predictor is None:
            raise RuntimeError("Model not initialized. Check checkpoint paths.")
        
        # Extract text label for GroundingDINO (remove "segment" prefix if present)
        clean_prompt = text_prompt.replace("segment ", "").strip()
        
        # Step 1: Detect with GroundingDINO
        detections = self.grounding_dino.predict_with_classes(
            image=image,
            classes=[clean_prompt],
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )
        
        # If no detections, return empty mask
        if len(detections.xyxy) == 0:
            empty_mask = np.zeros(image.shape[:2], dtype=np.float32)
            if return_boxes:
                return empty_mask, np.array([])
            return empty_mask
        
        # Step 2: Segment with SAM
        self.sam_predictor.set_image(image)
        
        # Convert boxes to SAM format
        boxes = detections.xyxy
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            torch.tensor(boxes, device=self.device),
            image.shape[:2]
        )
        
        # Predict masks for all boxes
        masks, scores, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
        )
        
        # Combine all masks (union)
        combined_mask = masks.any(dim=0).squeeze().cpu().numpy().astype(np.float32)
        
        if return_boxes:
            return combined_mask, boxes
        return combined_mask
    
    def train(self):
        """Set models to training mode"""
        # Note: For fine-tuning, we typically only fine-tune SAM decoder
        if self.sam_predictor:
            self.sam_predictor.model.train()
    
    def eval(self):
        """Set models to evaluation mode"""
        if self.sam_predictor:
            self.sam_predictor.model.eval()
    
    def to(self, device):
        """Move models to device"""
        self.device = device
        if self.sam_predictor:
            self.sam_predictor.model.to(device)
        return self


def download_model_checkpoints(cache_dir: str = "../checkpoints") -> dict:
    """
    Download pre-trained model checkpoints.
    
    Returns:
        Dictionary with paths to model checkpoints
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    import urllib.request
    
    checkpoints = {}
    
    # SAM checkpoint (ViT-H)
    sam_checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    sam_checkpoint_path = os.path.join(cache_dir, "sam_vit_h_4b8939.pth")
    
    if not os.path.exists(sam_checkpoint_path):
        print(f"Downloading SAM checkpoint to {sam_checkpoint_path}...")
        urllib.request.urlretrieve(sam_checkpoint_url, sam_checkpoint_path)
        print("SAM checkpoint downloaded!")
    
    checkpoints['sam'] = sam_checkpoint_path
    
    # GroundingDINO checkpoints are typically downloaded separately
    # Users should download from: https://github.com/IDEA-Research/GroundingDINO
    print("\nNote: GroundingDINO checkpoints should be downloaded separately from:")
    print("https://github.com/IDEA-Research/GroundingDINO")
    
    return checkpoints


if __name__ == "__main__":
    # Test the model
    print("Testing GroundedSAM model...")
    
    # Download checkpoints
    checkpoints = download_model_checkpoints()
    print(f"Checkpoints: {checkpoints}")

