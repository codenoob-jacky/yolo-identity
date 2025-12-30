#!/bin/bash

# Installation script for YOLO-based Construction Safety Detection System with 100-200 Classes

echo "Installing YOLO-based Construction Safety Detection System with 100-200 Classes"
echo "==============================================================================="

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "This script is designed for Linux systems. Please install dependencies manually for your OS."
    exit 1
fi

# Check if Python 3.7+ is installed
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo "Found Python: $PYTHON_VERSION"
    
    # Extract major and minor version numbers
    IFS='.' read -ra VERSION_PARTS <<< "$PYTHON_VERSION"
    MAJOR=${VERSION_PARTS[0]}
    MINOR=${VERSION_PARTS[1]}
    
    # Check if Python version is 3.7 or higher
    if [[ $MAJOR -eq 3 && $MINOR -ge 7 ]] || [[ $MAJOR -gt 3 ]]; then
        echo "Python version is sufficient"
    else
        echo "Error: Python 3.7 or higher is required"
        exit 1
    fi
else
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Create virtual environment if not already in one
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv safety_detection_env
    source safety_detection_env/bin/activate
    echo "Virtual environment created and activated."
else
    echo "Using existing virtual environment."
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

# Install additional dependencies that might be needed for 100-200 classes
echo "Installing additional dependencies for large-scale training..."
pip install opencv-python-headless>=4.5.0
pip install ultralytics>=8.0.0
pip install matplotlib>=3.2.0
pip install pandas>=1.1.0
pip install seaborn>=0.11.0
pip install scikit-learn>=0.24.0
pip install tensorboard>=2.10.0
pip install albumentations>=1.3.0  # For advanced data augmentation
pip install pyyaml>=6.0

# Check for CUDA availability and install appropriate PyTorch
if nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "No NVIDIA GPU detected. Installing PyTorch with CPU support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Verify installation of key packages
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
python -c "from ultralytics import YOLO; print('Ultralytics YOLO installed successfully')"

# Check CUDA availability if PyTorch is installed
python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA available: {torch.cuda.get_device_name(0)}')
else:
    print('CUDA not available, using CPU')
"

echo "Installation completed successfully!"
echo ""
echo "To train your model with 100-200 classes:"
echo "1. Prepare your dataset with 100-200 safety violation classes"
echo "2. Create dataset structure: python utils/data_preparation.py"
echo "3. Train your model: python src/train_safety_model.py --dataset_path /path/to/dataset --num_classes 150"
echo ""
echo "For more information, check IMPLEMENTATION_GUIDE.md"
echo "==============================================================================="