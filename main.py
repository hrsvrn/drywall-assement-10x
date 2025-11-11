#!/usr/bin/env python3
"""
Drywall QA - Complete Pipeline
Main entry point for preprocessing and training
"""

import os
import sys
import argparse


def run_preprocessing():
	"""Run all preprocessing scripts."""
	print("\n" + "="*60)
	print("PREPROCESSING PIPELINE")
	print("="*60)
	
	root_dir = os.path.dirname(os.path.abspath(__file__))
	scripts_dir = os.path.join(root_dir, 'scripts')
	
	# Download datasets
	print("\n[1/3] Downloading datasets from Roboflow...")
	result = os.system(f"cd {root_dir} && python3 {os.path.join(scripts_dir, 'download_script_roboflow.py')}")
	if result != 0:
		print("ERROR: Failed to download datasets. Make sure ROBOFLOW_API_KEY is set.")
		return False
	
	# Convert COCO to masks
	print("\n[2/3] Converting COCO annotations to binary masks...")
	result = os.system(f"cd {root_dir} && python3 {os.path.join(scripts_dir, 'coco_to_masks.py')}")
	if result != 0:
		print("ERROR: Failed to convert annotations.")
		return False
	
	# Generate metadata CSV
	print("\n[3/3] Generating metadata CSV...")
	result = os.system(f"cd {root_dir} && python3 {os.path.join(scripts_dir, 'generate_metadata.py')}")
	if result != 0:
		print("ERROR: Failed to generate metadata.")
		return False
	
	print("\nPreprocessing complete!")
	return True


def run_training():
	"""Run model training."""
	print("\n" + "="*60)
	print("MODEL TRAINING")
	print("="*60)
	
	root_dir = os.path.dirname(os.path.abspath(__file__))
	src_dir = os.path.join(root_dir, 'src')
	
	# Check if dataset CSV exists
	csv_path = os.path.join(root_dir, 'processed_datasets', 'dataset.csv')
	if not os.path.exists(csv_path):
		print(f"ERROR: Dataset CSV not found: {csv_path}")
		print("Please run preprocessing first: python3 main.py --mode preprocess")
		return False
	
	print(f"\nDataset CSV found: {csv_path}")
	print("Starting training...\n")
	
	result = os.system(f"cd {src_dir} && python3 main.py")
	
	if result != 0:
		print("\nERROR: Training failed.")
		return False
	
	print("\nTraining complete!")
	return True


def main():
	parser = argparse.ArgumentParser(
		description="Drywall QA - Prompted Segmentation Pipeline",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  # Run complete pipeline (preprocessing + training)
  python3 main.py --mode all
  
  # Only preprocessing
  python3 main.py --mode preprocess
  
  # Only training (default)
  python3 main.py --mode train
  python3 main.py
		"""
	)
	
	parser.add_argument(
		'--mode',
		choices=['all', 'preprocess', 'train'],
		default='train',
		help='Pipeline mode (default: train)'
	)
	
	args = parser.parse_args()
	
	print("\n" + "="*60)
	print("DRYWALL QA - PROMPTED SEGMENTATION")
	print("="*60)
	print(f"Mode: {args.mode.upper()}")
	
	success = True
	
	# Run preprocessing if requested
	if args.mode in ['all', 'preprocess']:
		if not run_preprocessing():
			success = False
			if args.mode == 'all':
				print("\nERROR: Stopping pipeline due to preprocessing failure.")
				sys.exit(1)
	
	# Run training if requested
	if args.mode in ['all', 'train']:
		if not run_training():
			success = False
	
	print("\n" + "="*60)
	if success:
		print("PIPELINE COMPLETED SUCCESSFULLY!")
	else:
		print("PIPELINE COMPLETED WITH ERRORS")
	print("="*60 + "\n")
	
	sys.exit(0 if success else 1)


if __name__ == "__main__":
	main()

