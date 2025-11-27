# F5-TTS API Ubuntu Installation Guide

## Option 1: Complete Automated Setup (Recommended)

### Single Command Installation
```bash
# Download and run complete setup
curl -sSL https://raw.githubusercontent.com/SWivid/F5-TTS/main/complete_setup.sh | bash

# OR if you have the files locally:
chmod +x complete_setup.sh
sudo ./complete_setup.sh
```

This will:
- ✅ Install all system dependencies
- ✅ Install Miniconda automatically  
- ✅ Create conda environment
- ✅ Install PyTorch with CUDA 12.4
- ✅ Install F5-TTS and API dependencies
- ✅ Create startup scripts
- ✅ Create systemd service
- ✅ Test everything

## Option 2: Manual Installation

### Step 1: Install Miniconda
```bash
# Download and install Miniconda
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
sudo bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3

# Add to PATH
echo 'export PATH="/opt/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Initialize conda
/opt/miniconda3/bin/conda init bash
source ~/.bashrc
```

### Step 2: Install System Dependencies
```bash
sudo apt update
sudo apt install -y wget curl git ffmpeg libsndfile1 portaudio19-dev python3-dev build-essential
```

### Step 3: Create Environment and Install
```bash
# Create environment
conda create -n f5-tts python=3.11 -y

# Install PyTorch with CUDA
conda run -n f5-tts pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

# Install F5-TTS and API
conda run -n f5-tts pip install -e .
conda run -n f5-tts pip install -r api_requirements_linux.txt
```

## Running the Server

### Quick Start
```bash
./start_f5tts_api.sh
```

### As System Service
```bash
# Enable and start service
sudo systemctl enable f5tts-api
sudo systemctl start f5tts-api

# Check status
sudo systemctl status f5tts-api

# View logs
sudo journalctl -u f5tts-api -f
```

### Manual Start
```bash
conda activate f5-tts
python f5_tts_api.py --host 0.0.0.0 --port 8000
```

## Access Your API

- **API Server**: `http://YOUR_SERVER_IP:8000`
- **Swagger Docs**: `http://YOUR_SERVER_IP:8000/docs`  
- **Health Check**: `http://YOUR_SERVER_IP:8000/health`

## Test Installation
```bash
# Test API health
curl http://localhost:8000/health

# Test GPU detection
conda activate f5-tts
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

## Troubleshooting

### If conda command not found:
```bash
export PATH="/opt/miniconda3/bin:$PATH"
source /opt/miniconda3/etc/profile.d/conda.sh
```

### If CUDA issues:
```bash
# Check NVIDIA setup
nvidia-smi
nvcc --version

# Reinstall PyTorch
conda activate f5-tts
pip uninstall torch torchaudio
pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
```

Your RTX 4090 setup will be perfect for F5-TTS! 🚀