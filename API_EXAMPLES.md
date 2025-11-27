# F5-TTS API Advanced Settings Examples

## 🎯 Basic Usage (Uses Gradio defaults)
```bash
curl -X POST "http://localhost:8000/tts-generate" \
     -H "Content-Type: application/json" \
     -d '{
       "audio_file_id": "your-file-id", 
       "text": "Hello, this will use all default Gradio settings!"
     }'
```

## ⚙️ Advanced Usage (Custom settings)
```bash
curl -X POST "http://localhost:8000/tts-generate" \
     -H "Content-Type: application/json" \
     -d '{
       "audio_file_id": "your-file-id",
       "text": "This uses custom settings!",
       "ref_text": "Optional custom reference text",
       "settings": {
         "randomize_seed": false,
         "seed": 12345,
         "remove_silence": true,
         "speed": 1.2,
         "nfe_step": 24,
         "cross_fade_duration": 0.3
       }
     }'
```

## 🎛️ All Available Settings

| Setting | Type | Range | Default | Description |
|---------|------|-------|---------|-------------|
| `randomize_seed` | bool | - | true | Use random seed for each generation |
| `seed` | int | 0-2147483647 | 0 | Specific seed for reproducible results |
| `remove_silence` | bool | - | false | Auto-detect and crop long silences |
| `speed` | float | 0.3-2.0 | 1.0 | Audio playback speed multiplier |
| `nfe_step` | int | 4-64 | 32 | Number of denoising steps (higher = better quality, slower) |
| `cross_fade_duration` | float | 0.0-1.0 | 0.15 | Cross-fade duration between audio clips in seconds |

## 🔍 Check Available Settings
```bash
curl http://localhost:8000/
```

Returns all default settings and parameter descriptions.

## 📝 Python Example
```python
import requests

# Upload audio
with open("reference.wav", "rb") as f:
    response = requests.post("http://localhost:8000/upload-audio", files={"audio_file": f})
file_id = response.json()["file_id"]

# Generate with custom settings
response = requests.post(
    "http://localhost:8000/tts-generate",
    json={
        "audio_file_id": file_id,
        "text": "This is a test with custom settings!",
        "settings": {
            "randomize_seed": False,
            "seed": 42,
            "remove_silence": True,
            "speed": 1.1,
            "nfe_step": 28,
            "cross_fade_duration": 0.2
        }
    }
)

result = response.json()
output_id = result["audio_file_id"]
seed_used = result["seed_used"]

# Download result
response = requests.get(f"http://localhost:8000/download-audio/{output_id}")
with open("output.wav", "wb") as f:
    f.write(response.content)

print(f"Generated with seed: {seed_used}")
```

Perfect match to Gradio interface! 🎵