import os
import shutil
from roboflow import Roboflow
from dotenv import load_dotenv

# Get the root directory (parent of scripts folder)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(root_dir)  # Change to root directory for consistent paths

load_dotenv()  # loads project .env into os.environ
# Read the Roboflow API key from an environment variable to avoid hard-coding secrets.
# Export it before running this script, for example:
#   export ROBOFLOW_API_KEY="your_key_here"
api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
	raise RuntimeError(
		"Environment variable ROBOFLOW_API_KEY is not set. "
		"Set it before running this script, e.g.:\n"
		"  export ROBOFLOW_API_KEY='your_key_here'"
	)

rf = Roboflow(api_key=api_key)

# Download first dataset
project = rf.workspace("hrsvrn").project("cracks-3ii36-xl6wn")
version = project.version(1)
dataset = version.download("coco")
# Move to dataset folder
if os.path.exists(dataset.location):
	dataset_folder = os.path.join(root_dir, "dataset")
	os.makedirs(dataset_folder, exist_ok=True)
	destination = os.path.join(dataset_folder, os.path.basename(dataset.location))
	if os.path.exists(destination):
		shutil.rmtree(destination)
	shutil.move(dataset.location, destination)
	print(f"Moved {dataset.location} to {destination}")

# Download second dataset
project = rf.workspace("hrsvrn").project("drywall-join-detect-se2uo")
version = project.version(1)
dataset = version.download("coco")
# Move to dataset folder
if os.path.exists(dataset.location):
	dataset_folder = os.path.join(root_dir, "dataset")
	os.makedirs(dataset_folder, exist_ok=True)
	destination = os.path.join(dataset_folder, os.path.basename(dataset.location))
	if os.path.exists(destination):
		shutil.rmtree(destination)
	shutil.move(dataset.location, destination)
	print(f"Moved {dataset.location} to {destination}")

