import os
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


def create_binary_mask_from_polygons(polygons, image_shape):
	# Convert COCO polygon annotations to binary mask.
	
	# Args:
	# 	polygons: List of polygon coordinates [x1,y1,x2,y2,...]
	# 	image_shape: Tuple of (height, width)
	
	# Returns:
	# 	Binary mask as numpy array
	mask = Image.new('L', (image_shape[1], image_shape[0]), 0)
	draw = ImageDraw.Draw(mask)
	
	# Convert flat list to list of tuples [(x1,y1), (x2,y2), ...]
	for polygon in polygons:
		if len(polygon) >= 6:  # Need at least 3 points
			coords = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
			draw.polygon(coords, outline=255, fill=255)
	
	return np.array(mask)


def process_coco_dataset(dataset_path, output_path, dataset_name, split):
	"""
	Process a COCO dataset split and convert to binary masks.
	
	Args:
		dataset_path: Path to the dataset folder (e.g., 'dataset/cracks-1')
		output_path: Path to output folder (e.g., 'processed_datasets')
		dataset_name: Name of the dataset (e.g., 'cracks-1')
		split: Dataset split ('train' or 'valid')
	"""
	# Paths
	split_path = os.path.join(dataset_path, split)
	annotation_file = os.path.join(split_path, '_annotations.coco.json')
	
	# Check if annotation file exists
	if not os.path.exists(annotation_file):
		print(f"Warning: Annotation file not found: {annotation_file}")
		return
	
	# Create output directories
	output_images_dir = os.path.join(output_path, dataset_name, split, 'images')
	output_masks_dir = os.path.join(output_path, dataset_name, split, 'masks')
	os.makedirs(output_images_dir, exist_ok=True)
	os.makedirs(output_masks_dir, exist_ok=True)
	
	# Load COCO annotations
	print(f"\nProcessing {dataset_name}/{split}...")
	with open(annotation_file, 'r') as f:
		coco_data = json.load(f)
	
	# Create image_id to filename mapping
	image_info = {img['id']: img for img in coco_data['images']}
	
	# Group annotations by image_id
	annotations_by_image = {}
	for ann in coco_data['annotations']:
		image_id = ann['image_id']
		if image_id not in annotations_by_image:
			annotations_by_image[image_id] = []
		annotations_by_image[image_id].append(ann)
	
	# Process each image
	processed_count = 0
	skipped_count = 0
	
	for img_id, img_data in tqdm(image_info.items(), desc=f"{dataset_name}/{split}"):
		filename = img_data['file_name']
		image_path = os.path.join(split_path, filename)
		
		# Check if image exists
		if not os.path.exists(image_path):
			print(f"Warning: Image not found: {image_path}")
			skipped_count += 1
			continue
		
		# Copy image to output directory
		output_image_path = os.path.join(output_images_dir, filename)
		shutil.copy2(image_path, output_image_path)
		
		# Create binary mask
		height = img_data['height']
		width = img_data['width']
		combined_mask = np.zeros((height, width), dtype=np.uint8)
		
		# Get annotations for this image
		if img_id in annotations_by_image:
			for ann in annotations_by_image[img_id]:
				if 'segmentation' in ann and ann['segmentation']:
					# Handle both list of polygons and single polygon
					segmentation = ann['segmentation']
					if isinstance(segmentation, list):
						# Check if it's a list of polygons or a single polygon
						if isinstance(segmentation[0], list):
							# Multiple polygons
							mask = create_binary_mask_from_polygons(segmentation, (height, width))
						else:
							# Single polygon as flat list
							mask = create_binary_mask_from_polygons([segmentation], (height, width))
						
						# Combine masks (union)
						combined_mask = np.maximum(combined_mask, mask)
		
		# Save mask
		mask_filename = os.path.splitext(filename)[0] + '_mask.png'
		mask_path = os.path.join(output_masks_dir, mask_filename)
		Image.fromarray(combined_mask).save(mask_path)
		
		processed_count += 1
	
	print(f"Processed {processed_count} images, skipped {skipped_count} images")
	
	# Save dataset info
	info = {
		'dataset_name': dataset_name,
		'split': split,
		'num_images': processed_count,
		'num_categories': len(coco_data.get('categories', [])),
		'categories': coco_data.get('categories', []),
		'original_annotation_file': annotation_file
	}
	
	info_file = os.path.join(output_path, dataset_name, split, 'dataset_info.json')
	with open(info_file, 'w') as f:
		json.dump(info, f, indent=2)


def main():
	"""Main function to process all datasets."""
	# Define paths - get root directory (parent of scripts folder)
	root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	dataset_dir = os.path.join(root_dir, 'dataset')
	output_dir = os.path.join(root_dir, 'processed_datasets')
	
	# Create output directory
	os.makedirs(output_dir, exist_ok=True)
	print(f"Output directory: {output_dir}")
	
	# Define datasets to process
	datasets = ['cracks-1', 'Drywall-Join-Detect-1']
	splits = ['train', 'valid']
	
	# Process each dataset
	for dataset_name in datasets:
		dataset_path = os.path.join(dataset_dir, dataset_name)
		if not os.path.exists(dataset_path):
			print(f"Warning: Dataset not found: {dataset_path}")
			continue
		
		for split in splits:
			try:
				process_coco_dataset(dataset_path, output_dir, dataset_name, split)
			except Exception as e:
				print(f"Error processing {dataset_name}/{split}: {str(e)}")
				import traceback
				traceback.print_exc()
	
	print("\n" + "="*60)
	print("Processing complete!")
	print(f"Processed datasets saved to: {output_dir}")
	print("="*60)


if __name__ == "__main__":
	main()

