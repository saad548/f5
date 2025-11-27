#!/bin/bash
# Quick test to verify the API fix

echo "🔧 Testing F5-TTS API fix..."

# Restart the server in background
echo "🚀 Restarting server..."
pkill -f f5_tts_api.py
sleep 2

# Start server in background
cd "$(dirname "$0")"
source f5tts_env/bin/activate
nohup python3 f5_tts_api.py --host 0.0.0.0 --port 8000 > api_server.log 2>&1 &
SERVER_PID=$!

echo "🔍 Server started with PID: $SERVER_PID"
echo "📋 Waiting 10 seconds for server to load models..."
sleep 10

# Test health endpoint
echo "❤️ Testing health endpoint..."
curl -s http://localhost:8000/health | jq .

echo ""
echo "✅ Server should be ready now!"
echo "📊 Check logs: tail -f api_server.log"
echo "🌐 Access: http://YOUR_IP:8000/docs"
echo "🛑 Stop server: kill $SERVER_PID"