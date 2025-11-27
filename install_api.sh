#!/bin/bash
# F5-TTS API Server Installation Script for Ubuntu Linux
# This script sets up the F5-TTS API server with CUDA support for RTX 4090

set -e  # Exit on any error

echo "========================================"
echo "F5-TTS API Server Installation (Ubuntu)"
echo "========================================"

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Warning: Running as root. Consider using a regular user account."
fi

# Check for conda/miniconda
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda/Anaconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check NVIDIA driver and CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: NVIDIA drivers not found. Please install NVIDIA drivers first."
    exit 1
fi

echo "NVIDIA GPU detected:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Step 1: Create conda environment
echo "Step 1: Creating conda environment..."
conda create -n f5-tts python=3.11 -y

# Step 2: Activate environment
echo "Step 2: Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate f5-tts

# Step 3: Install PyTorch with CUDA 12.4 (compatible with your CUDA 12.9)
echo "Step 3: Installing PyTorch with CUDA support..."
pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

# Verify PyTorch CUDA installation
echo "Verifying PyTorch CUDA installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

if [ $? -ne 0 ]; then
    echo "Error: PyTorch CUDA installation verification failed"
    exit 1
fi

# Step 4: Install F5-TTS
echo "Step 4: Installing F5-TTS..."
pip install -e .

# Step 5: Install API requirements
echo "Step 5: Installing API requirements..."
pip install -r api_requirements.txt

# Step 6: Test model loading (optional but recommended)
echo "Step 6: Testing model loading..."
python -c "
try:
    from f5_tts.infer.utils_infer import load_vocoder, load_model
    from f5_tts.model import DiT
    import torch
    print('✅ F5-TTS imports successful')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'❌ Error: {e}')
    exit(1)
"

echo "========================================"
echo "Installation completed successfully!"
echo "========================================"
echo ""
echo "To start the API server:"
echo "1. Activate the environment: conda activate f5-tts"
echo "2. Run the server: python f5_tts_api.py"
echo ""
echo "Server options:"
echo "  --host 0.0.0.0          # Bind to all interfaces"
echo "  --port 8000             # Port number"
echo "  --reload                # Enable auto-reload for development"
echo ""
echo "The API will be available at: http://localhost:8000"
echo "Swagger documentation: http://localhost:8000/docs"
echo ""
echo "For production, consider using:"
echo "  nohup python f5_tts_api.py --host 0.0.0.0 --port 8000 > api.log 2>&1 &"