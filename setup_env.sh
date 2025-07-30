#!/bin/bash
# Setup script for the dynamics autoencoder workspace.

set -e  # Exit on any error

echo "Setting up Dynamics Autoencoder Workspace..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Found conda, creating conda environment..."
    ENV_TYPE="conda"
    ENV_NAME="dynamics-ae"
    
    # Create conda environment
    conda create -n $ENV_NAME python=3.9 -y
    
    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $ENV_NAME
    
    echo "Conda environment '$ENV_NAME' created and activated."
    
elif command -v python3 &> /dev/null; then
    echo "Using python3 venv..."
    ENV_TYPE="venv"
    ENV_NAME="dynamics-ae"
    
    # Create virtual environment
    python3 -m venv $ENV_NAME
    
    # Activate environment
    source $ENV_NAME/bin/activate
    
    echo "Virtual environment '$ENV_NAME' created and activated."
    
else
    echo "Error: Neither conda nor python3 found. Please install Python 3.9+ or Anaconda/Miniconda."
    exit 1
fi

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup completed successfully!"
echo ""
echo "To activate the environment in the future:"
if [ "$ENV_TYPE" = "conda" ]; then
    echo "  conda activate $ENV_NAME"
else
    echo "  source $ENV_NAME/bin/activate"
fi
echo ""
echo "To run the pipeline:"
echo "  python main.py --config configs/example_config.yaml"
echo ""
echo "To deactivate the environment:"
if [ "$ENV_TYPE" = "conda" ]; then
    echo "  conda deactivate"
else
    echo "  deactivate"
fi 