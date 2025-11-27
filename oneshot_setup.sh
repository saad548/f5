#!/bin/bash
# ONE-SHOT F5-TTS API SETUP AND RUN SCRIPT
# This script does EVERYTHING: installs, configures, and runs the F5-TTS API server

set -e  # Exit immediately on any error

echo "🚀 ONE-SHOT F5-TTS API SETUP - STARTING..."

# Install system dependencies
echo "📦 Installing system dependencies..."
apt update -y
apt install -y wget curl git ffmpeg libsndfile1 python3-pip python3-venv python3-dev build-essential

# Check GPU
echo "🎮 Checking GPU..."
nvidia-smi

# Create Python virtual environment (simpler than conda)
echo "🐍 Creating Python virtual environment..."
python3 -m venv f5tts_env
source f5tts_env/bin/activate

# Install PyTorch with CUDA
echo "🔥 Installing PyTorch with CUDA..."
pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

# Verify CUDA
echo "✅ Verifying CUDA..."
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Install F5-TTS (following official README method)
echo "🎵 Installing F5-TTS..."
pip install f5-tts

# Install API dependencies
echo "📡 Installing API dependencies..."
pip install fastapi uvicorn[standard] python-multipart librosa soundfile pydub cached-path transformers safetensors psutil

# Test F5-TTS installation
echo "🧪 Testing F5-TTS..."
python3 -c "
try:
    import f5_tts
    print('✅ F5-TTS imported successfully!')
    import torch
    if torch.cuda.is_available():
        print(f'✅ GPU Ready: {torch.cuda.get_device_name(0)}')
        print(f'✅ VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB')
    else:
        print('❌ CUDA not available')
        exit(1)
except Exception as e:
    print(f'❌ Error: {e}')
    exit(1)
"

# Create simple run script
echo "📝 Creating run script..."
cat > run_api.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source f5tts_env/bin/activate
echo "🚀 Starting F5-TTS API Server..."
echo "🎮 GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
echo ""
python3 f5_tts_api.py --host 0.0.0.0 --port 8000
EOF

chmod +x run_api.sh

echo ""
echo "🎉 SETUP COMPLETE!"
echo ""
echo "🚀 STARTING F5-TTS API SERVER NOW..."
echo ""

# Start the API server immediately
source f5tts_env/bin/activate
python3 f5_tts_api.py --host 0.0.0.0 --port 8000