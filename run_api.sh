#!/bin/bash

# F5-TTS API Server Startup Script
echo "🚀 Starting F5-TTS API Server..."

# Change to script directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "f5tts_env" ]; then
    echo "📦 Activating virtual environment..."
    source f5tts_env/bin/activate
fi

# Check GPU status
echo "🎮 GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

echo ""
echo "🎤 Starting F5-TTS API Server..."
echo "📖 API Docs will be at: http://YOUR_SERVER_IP:8000/docs"
echo "🔐 Admin Panel will be at: http://YOUR_SERVER_IP:8000/admin" 
echo "👤 Admin Login: yasirr548 / yasirr548AJSKD#D45s"
echo ""

# Start the server
python3 f5_tts_api.py --host 0.0.0.0 --port 8000