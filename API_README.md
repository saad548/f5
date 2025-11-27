# F5-TTS API Server

A FastAPI server for F5-TTS that provides REST endpoints with automatic reference text transcription using the same default settings as the Gradio interface.

## Features

- **Upload Audio Endpoint**: Upload reference audio files
- **TTS Generation Endpoint**: Generate speech using uploaded audio as reference
- **Automatic Transcription**: Uses Whisper to automatically transcribe reference audio if no text provided
- **Default Gradio Settings**: Uses the same inference parameters as the original Gradio interface
- **Swagger Documentation**: Interactive API documentation

## Quick Installation

### Prerequisites
- Windows with NVIDIA GPU
- Conda installed

### Installation Steps

1. **Run the installation script:**
   ```cmd
   install_api.bat
   ```

   This will:
   - Create a conda environment with Python 3.11
   - Install PyTorch with CUDA 12.4 support
   - Install F5-TTS in editable mode
   - Install all API dependencies

### Manual Installation

If you prefer manual installation:

1. **Create conda environment:**
   ```cmd
   conda create -n f5-tts python=3.11
   conda activate f5-tts
   ```

2. **Install PyTorch with CUDA:**
   ```cmd
   pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
   ```

3. **Install F5-TTS:**
   ```cmd
   pip install -e .
   ```

4. **Install API requirements:**
   ```cmd
   pip install -r api_requirements.txt
   ```

## Running the Server

1. **Activate environment:**
   ```cmd
   conda activate f5-tts
   ```

2. **Start the server:**
   ```cmd
   python f5_tts_api.py
   ```

3. **Access the API:**
   - Server: `http://localhost:8000`
   - Swagger docs: `http://localhost:8000/docs`
   - Health check: `http://localhost:8000/health`

## API Endpoints

### 1. Upload Audio
**POST** `/upload-audio`

Upload a reference audio file for TTS generation.

- **Input**: Audio file (WAV, MP3, etc.)
- **Output**: File ID for later use
- **Max duration**: 12 seconds recommended

### 2. Generate TTS
**POST** `/tts-generate`

Generate TTS audio using uploaded reference audio.

**Request body:**
```json
{
  "audio_file_id": "uuid-from-upload",
  "text": "Text to generate speech for",
  "ref_text": "Optional reference text (auto-transcribed if empty)",
  "settings": {
    "remove_silence": false,
    "cross_fade_duration": 0.15,
    "nfe_step": 32,
    "speed": 1.0,
    "seed": -1
  }
}
```

**Response:**
```json
{
  "audio_file_id": "uuid-for-download",
  "ref_text": "Transcribed or provided reference text",
  "message": "TTS audio generated successfully",
  "seed_used": 12345
}
```

### 3. Download Audio
**GET** `/download-audio/{file_id}`

Download generated audio file.

### Additional Endpoints
- **GET** `/health` - Health check
- **GET** `/list-audio` - List uploaded files
- **DELETE** `/audio/{file_id}` - Delete audio file

## Default Settings

The API uses the same default settings as the Gradio interface:

- **Model**: F5-TTS v1 Base
- **Remove Silence**: false
- **Cross-fade Duration**: 0.15s
- **NFE Steps**: 32
- **Speed**: 1.0
- **Seed**: Random (-1)

## Example Usage

### Using curl:

1. **Upload audio:**
   ```bash
   curl -X POST "http://localhost:8000/upload-audio" \
        -F "audio_file=@reference.wav"
   ```

2. **Generate TTS:**
   ```bash
   curl -X POST "http://localhost:8000/tts-generate" \
        -H "Content-Type: application/json" \
        -d '{
          "audio_file_id": "your-file-id",
          "text": "Hello, this is a test of the F5-TTS API!"
        }'
   ```

3. **Download result:**
   ```bash
   curl "http://localhost:8000/download-audio/generated-file-id" \
        -o output.wav
   ```

### Using Python requests:

```python
import requests

# Upload audio
with open("reference.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload-audio",
        files={"audio_file": f}
    )
file_id = response.json()["file_id"]

# Generate TTS
response = requests.post(
    "http://localhost:8000/tts-generate",
    json={
        "audio_file_id": file_id,
        "text": "Your text here"
    }
)
output_id = response.json()["audio_file_id"]

# Download result
response = requests.get(f"http://localhost:8000/download-audio/{output_id}")
with open("output.wav", "wb") as f:
    f.write(response.content)
```

## Notes

- Audio files are automatically cleaned up after 1 hour
- Maximum recommended reference audio length: 12 seconds
- The server automatically transcribes reference audio using Whisper if no reference text is provided
- All inference uses the F5-TTS v1 model with the same quality settings as Gradio