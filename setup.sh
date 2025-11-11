#!/bin/bash

echo "================================================"
echo "Drywall QA - Setup Script"
echo "================================================"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p dataset
mkdir -p processed_datasets
mkdir -p checkpoints
mkdir -p results

# Check if .env file exists
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file..."
    echo "# Roboflow API Key" > .env
    echo "# Get yours at: https://roboflow.com/ → Settings → API Keys" >> .env
    echo "ROBOFLOW_API_KEY=your_api_key_here" >> .env
    echo ""
    echo "WARNING: Please edit .env and add your ROBOFLOW_API_KEY"
fi

# Check Python version
echo ""
echo "Checking Python version..."
python3 --version

# Install requirements
echo ""
read -p "Install Python requirements? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing requirements..."
    pip3 install -r requirements.txt
fi

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your ROBOFLOW_API_KEY"
echo "2. Run: python3 main.py --mode all"
echo ""

