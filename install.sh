#!/bin/bash

# Installation script for Construction Site Safety Detection System

echo "Installing Construction Site Safety Detection System..."
echo "====================================================="

# Check if Python 3.8+ is installed
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo "Found Python: $PYTHON_VERSION"
    
    # Check if Python version is 3.8 or higher
    if [[ $(printf '%s\n' "3.8" "$PYTHON_VERSION" | sort -V | head -n1) == "3.8" ]]; then
        echo "Python version is sufficient"
    else
        echo "Error: Python 3.8 or higher is required"
        exit 1
    fi
else
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Install pip if not installed
if ! command -v pip &> /dev/null; then
    echo "Installing pip..."
    python3 -m ensurepip --upgrade
fi

# Install required packages
echo "Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation of key packages
echo "Verifying installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
python3 -c "from ultralytics import YOLO; print('Ultralytics YOLO installed successfully')"

echo "Installation completed successfully!"
echo ""
echo "To get started:"
echo "1. Prepare your safety equipment dataset"
echo "2. Train your model: python src/train_safety_model.py"
echo "3. Run inference: python -m src.detection --input webcam"
echo ""
echo "For more information, check IMPLEMENTATION_GUIDE.md"