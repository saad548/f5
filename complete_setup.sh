#!/bin/bash
# Complete F5-TTS API Server Setup Script for Ubuntu Linux
# This script installs everything from scratch including Miniconda

set -e  # Exit on any error

echo "=========================================="
echo "Complete F5-TTS API Setup (Ubuntu 22.04)"
echo "=========================================="

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Warning: Running as root. This will install conda system-wide."
    echo "Press Ctrl+C to cancel, or Enter to continue..."
    read
fi

# Update system packages
echo "Step 1: Updating system packages..."
apt update && apt upgrade -y

# Install system dependencies
echo "Step 2: Installing system dependencies..."
apt install -y wget curl git ffmpeg libsndfile1 portaudio19-dev python3-dev build-essential

# Check NVIDIA driver and CUDA
echo "Step 3: Checking NVIDIA setup..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: NVIDIA drivers not found. Please install NVIDIA drivers first."
    echo "Run: ubuntu-drivers devices && ubuntu-drivers autoinstall"
    exit 1
fi

echo "NVIDIA GPU detected:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Install Miniconda if not present
if ! command -v conda &> /dev/null; then
    echo "Step 4: Installing Miniconda..."
    cd /tmp
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b -p /opt/miniconda3
    
    # Add conda to PATH
    echo 'export PATH="/opt/miniconda3/bin:$PATH"' >> ~/.bashrc
    export PATH="/opt/miniconda3/bin:$PATH"
    
    # Initialize conda
    /opt/miniconda3/bin/conda init bash
    source ~/.bashrc
    
    echo "✅ Miniconda installed successfully!"
else
    echo "Step 4: Conda already installed, skipping..."
fi

# Source conda
if [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
    source /opt/miniconda3/etc/profile.d/conda.sh
elif [ -f "~/miniconda3/etc/profile.d/conda.sh" ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
fi

# Create conda environment
echo "Step 5: Creating F5-TTS conda environment..."
conda create -n f5-tts python=3.11 -y

# Activate environment
echo "Step 6: Activating environment..."
conda activate f5-tts

# Install PyTorch with CUDA
echo "Step 7: Installing PyTorch with CUDA support..."
conda run -n f5-tts pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

# Verify PyTorch CUDA installation
echo "Step 8: Verifying PyTorch CUDA installation..."
conda run -n f5-tts python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"

if [ $? -ne 0 ]; then
    echo "❌ Error: PyTorch CUDA installation verification failed"
    exit 1
fi

# Install F5-TTS
echo "Step 9: Installing F5-TTS..."
conda run -n f5-tts pip install -e .

# Install API requirements
echo "Step 10: Installing API requirements..."
conda run -n f5-tts pip install -r api_requirements_linux.txt

# Test F5-TTS loading
echo "Step 11: Testing F5-TTS model loading..."
conda run -n f5-tts python -c "
try:
    from f5_tts.infer.utils_infer import load_vocoder, load_model
    from f5_tts.model import DiT
    import torch
    print('✅ F5-TTS imports successful')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
        print(f'✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB')
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
"

# Create startup script
echo "Step 12: Creating startup script..."
cat > start_f5tts_api.sh << 'EOF'
#!/bin/bash
# F5-TTS API Startup Script

# Source conda
if [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
    source /opt/miniconda3/etc/profile.d/conda.sh
elif [ -f "~/miniconda3/etc/profile.d/conda.sh" ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
fi

# Activate environment and start API
conda activate f5-tts
echo "Starting F5-TTS API server..."
python f5_tts_api.py --host 0.0.0.0 --port 8000 "$@"
EOF

chmod +x start_f5tts_api.sh

# Create systemd service (optional)
echo "Step 13: Creating systemd service..."
cat > /etc/systemd/system/f5tts-api.service << EOF
[Unit]
Description=F5-TTS API Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/start_f5tts_api.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload

echo "============================================"
echo "✅ Installation completed successfully!"
echo "============================================"
echo ""
echo "🚀 To start the API server:"
echo "   ./start_f5tts_api.sh"
echo ""
echo "🔄 To start as system service:"
echo "   systemctl enable f5tts-api"
echo "   systemctl start f5tts-api"
echo "   systemctl status f5tts-api"
echo ""
echo "📊 To monitor the service:"
echo "   journalctl -u f5tts-api -f"
echo ""
echo "🌐 API will be available at:"
echo "   http://$(hostname -I | awk '{print $1}'):8000"
echo "   http://localhost:8000 (if local)"
echo ""
echo "📚 Swagger documentation:"
echo "   http://$(hostname -I | awk '{print $1}'):8000/docs"
echo ""
echo "🔍 Test the installation:"
echo "   curl http://localhost:8000/health"
echo ""
echo "💡 Your RTX 4090 is ready for F5-TTS! 🎵"