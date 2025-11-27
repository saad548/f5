# Quick Setup Commands for Ubuntu 22.04 with RTX 4090

## Prerequisites Check
```bash
# Check your system
lsb_release -a
nvidia-smi
```

## System Dependencies (if needed)
```bash
sudo apt update
sudo apt install -y ffmpeg libsndfile1 portaudio19-dev python3-dev git curl
```

## Quick Installation (One Command)
```bash
# Make script executable and run
chmod +x install_api.sh
./install_api.sh
```

## Manual Installation Steps

### 1. Create Environment
```bash
conda create -n f5-tts python=3.11 -y
conda activate f5-tts
```

### 2. Install PyTorch with CUDA
```bash
# PyTorch 2.4.0 with CUDA 12.4 (compatible with your CUDA 12.9)
pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
```

### 3. Verify CUDA Installation
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### 4. Install F5-TTS
```bash
pip install -e .
```

### 5. Install API Dependencies
```bash
pip install -r api_requirements_linux.txt
```

## Running the Server

### Development Mode
```bash
conda activate f5-tts
python f5_tts_api.py --reload
```

### Production Mode
```bash
conda activate f5-tts
nohup python f5_tts_api.py --host 0.0.0.0 --port 8000 > api.log 2>&1 &
```

### Check Server Status
```bash
# View logs
tail -f api.log

# Check if running
ps aux | grep f5_tts_api

# Test API
curl http://localhost:8000/health
```

## API Access
- **Server**: `http://your-server-ip:8000`
- **Swagger Docs**: `http://your-server-ip:8000/docs`
- **Health Check**: `http://your-server-ip:8000/health`

## Firewall (if needed)
```bash
# Open port 8000
sudo ufw allow 8000
sudo ufw status
```

## Example Usage
```bash
# Test upload
curl -X POST "http://localhost:8000/upload-audio" -F "audio_file=@reference.wav"

# Test TTS generation
curl -X POST "http://localhost:8000/tts-generate" \
     -H "Content-Type: application/json" \
     -d '{"audio_file_id": "your-file-id", "text": "Hello from Ubuntu server!"}'
```

## Troubleshooting

### CUDA Issues
```bash
# Check CUDA version
nvcc --version
nvidia-smi

# Reinstall PyTorch if needed
pip uninstall torch torchaudio
pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
```

### Memory Issues
```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# Clear GPU cache in Python
python -c "import torch; torch.cuda.empty_cache()"
```

### Audio Issues
```bash
# Install audio libraries
sudo apt install -y ffmpeg libsndfile1 portaudio19-dev

# Test audio processing
python -c "import librosa, soundfile; print('Audio libraries OK')"
```

Your RTX 4090 with 24GB VRAM is perfect for F5-TTS! 🚀