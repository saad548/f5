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
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
echo ""
python f5_tts_api.py --host 0.0.0.0 --port 8000 "$@"