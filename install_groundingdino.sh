#!/bin/bash
# Install GroundingDINO from source (more reliable than groundingdino-py)

echo "Installing GroundingDINO from source..."

# Clone repository
if [ ! -d "GroundingDINO" ]; then
    git clone https://github.com/IDEA-Research/GroundingDINO.git
    cd GroundingDINO
else
    cd GroundingDINO
    git pull
fi

# Install
pip install -e .

cd ..

echo "✅ GroundingDINO installed successfully!"
echo ""
echo "Next steps:"
echo "  1. Download model checkpoints: python download_models.py"
echo "  2. Preprocess data: python main.py --mode preprocess"
echo "  3. Train: python main.py --mode train"

