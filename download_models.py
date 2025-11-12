#!/usr/bin/env python3
"""
Script to download GroundingDINO and SAM model checkpoints.
Run this before training to set up all required model weights.
"""

import os
import sys
import urllib.request
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar"""
    print(f"Downloading from {url}")
    print(f"Saving to {output_path}")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    
    print(f"✓ Downloaded: {output_path}\n")


def main():
    # Create checkpoints directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("GroundingDINO + SAM Model Checkpoint Downloader")
    print("=" * 70)
    print()
    
    # SAM checkpoint
    sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    sam_path = checkpoint_dir / "sam_vit_h_4b8939.pth"
    
    if sam_path.exists():
        print(f"✓ SAM checkpoint already exists: {sam_path}")
    else:
        print("[1/3] Downloading SAM ViT-H checkpoint (~2.4 GB)...")
        try:
            download_url(sam_url, str(sam_path))
        except Exception as e:
            print(f"✗ Error downloading SAM: {e}")
            sys.exit(1)
    
    # GroundingDINO checkpoint
    gdino_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    gdino_path = checkpoint_dir / "groundingdino_swint_ogc.pth"
    
    if gdino_path.exists():
        print(f"✓ GroundingDINO checkpoint already exists: {gdino_path}")
    else:
        print("[2/3] Downloading GroundingDINO checkpoint (~694 MB)...")
        try:
            download_url(gdino_url, str(gdino_path))
        except Exception as e:
            print(f"✗ Error downloading GroundingDINO: {e}")
            sys.exit(1)
    
    # GroundingDINO config
    config_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    config_path = checkpoint_dir / "GroundingDINO_SwinT_OGC.py"
    
    if config_path.exists():
        print(f"✓ GroundingDINO config already exists: {config_path}")
    else:
        print("[3/3] Downloading GroundingDINO config...")
        try:
            download_url(config_url, str(config_path))
        except Exception as e:
            print(f"✗ Error downloading GroundingDINO config: {e}")
            sys.exit(1)
    
    print()
    print("=" * 70)
    print("✅ All model checkpoints downloaded successfully!")
    print("=" * 70)
    print()
    print("Downloaded files:")
    print(f"  - {sam_path} ({sam_path.stat().st_size / 1e9:.2f} GB)")
    print(f"  - {gdino_path} ({gdino_path.stat().st_size / 1e9:.2f} GB)")
    print(f"  - {config_path} ({config_path.stat().st_size / 1e3:.2f} KB)")
    print()
    print("Next steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Prepare datasets: python main.py --mode preprocess")
    print("  3. Train model: python src/train_grounded_sam.py")
    print()


if __name__ == "__main__":
    main()

