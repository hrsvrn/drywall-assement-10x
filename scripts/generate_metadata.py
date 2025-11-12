import csv
import glob
import os
from pathlib import Path

# Get the root directory (parent of scripts folder)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_datasets_dir = os.path.join(root_dir, "processed_datasets")
csv_path = os.path.join(processed_datasets_dir, "dataset.csv")

# Ensure the processed_datasets directory exists
os.makedirs(processed_datasets_dir, exist_ok=True)

# Check if file exists
file_exists = os.path.exists(csv_path)

# Collect existing entries (avoid duplicates)
existing_entries = set()
if file_exists:
    with open(csv_path, "r") as check_file:
        for row in csv.reader(check_file):
            if row and row[0] != "image_path":
                existing_entries.add(row[0])

# Open in append mode ('a') if exists, else write mode ('w')
with open(csv_path, "a" if file_exists else "w", newline='') as f:
    writer = csv.writer(f)

    # Write header only if new file
    if not file_exists:
        writer.writerow(["image_path", "mask_path", "prompt", "split"])

    def add_entries(dataset_name, split, prompt):
        """Add entries for a specific dataset and split."""
        images_dir = os.path.join(processed_datasets_dir, dataset_name, split, "images")
        
        if not os.path.exists(images_dir):
            print(f"Warning: Images directory not found: {images_dir}")
            return 0
        
        count = 0
        for img_path in glob.glob(os.path.join(images_dir, "*.jpg")):
            # Create corresponding mask path
            img_filename = os.path.basename(img_path)
            mask_filename = os.path.splitext(img_filename)[0] + "_mask.png"
            mask_path = os.path.join(processed_datasets_dir, dataset_name, split, "masks", mask_filename)
            
            # Check if mask exists and entry is not duplicate
            if os.path.exists(mask_path) and img_path not in existing_entries:
                # Convert to relative paths from root directory
                img_path_rel = os.path.relpath(img_path, root_dir)
                mask_path_rel = os.path.relpath(mask_path, root_dir)
                
                writer.writerow([img_path_rel, mask_path_rel, prompt, split])
                count += 1
        
        return count

    # Process both datasets with train, valid, and test splits
    total_count = 0
    
    # Cracks dataset
    print("Processing cracks-1 dataset...")
    count = add_entries("cracks-1", "train", "segment crack")
    print(f"  Added {count} training images")
    total_count += count
    
    count = add_entries("cracks-1", "valid", "segment crack")
    print(f"  Added {count} validation images")
    total_count += count
    
    count = add_entries("cracks-1", "test", "segment crack")
    print(f"  Added {count} test images")
    total_count += count
    
    # Drywall joints dataset (using "segment taping area" for consistency)
    print("Processing Drywall-Join-Detect-1 dataset...")
    count = add_entries("Drywall-Join-Detect-1", "train", "segment taping area")
    print(f"  Added {count} training images")
    total_count += count
    
    count = add_entries("Drywall-Join-Detect-1", "valid", "segment taping area")
    print(f"  Added {count} validation images")
    total_count += count
    
    count = add_entries("Drywall-Join-Detect-1", "test", "segment taping area")
    print(f"  Added {count} test images")
    total_count += count

print(f"\n{'='*60}")
print(f"dataset.csv updated at: {csv_path}")
print(f"Total new entries added: {total_count}")
print(f"{'='*60}")
