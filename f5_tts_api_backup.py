#!/usr/bin/env python3
"""
F5-TTS FastAPI Server
Provides REST API endpoints for F5-TTS model inference with automatic reference text transcription.
"""

import os
import tempfile
import uuid
import shutil
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import F5-TTS modules
from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiT
from cached_path import cached_path

# Models and configuration (from Gradio default settings)
DEFAULT_TTS_MODEL = "F5-TTS_v1"
DEFAULT_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]

# Default Gradio inference settings (exact match from Gradio interface)
DEFAULT_INFERENCE_SETTINGS = {
    "randomize_seed": True,
    "seed": 0,
    "remove_silence": False,
    "speed": 1.0,
    "nfe_step": 32,
    "cross_fade_duration": 0.15,
}

app = FastAPI(
    title="F5-TTS API",
    description="REST API for F5-TTS Text-to-Speech model with automatic reference text transcription",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model state
vocoder = None
F5TTS_ema_model = None
uploaded_files: Dict[str, str] = {}  # file_id -> file_path mapping

# Permanent storage for reference voices
REFERENCE_VOICES_DIR = "reference_voices"
os.makedirs(REFERENCE_VOICES_DIR, exist_ok=True)

# Pydantic models for request/response
class AudioUploadResponse(BaseModel):
    file_id: str
    message: str
    duration: Optional[float] = None

class TTSAdvancedSettings(BaseModel):
    randomize_seed: Optional[bool] = True
    seed: Optional[int] = 0
    remove_silence: Optional[bool] = False
    speed: Optional[float] = 1.0
    nfe_step: Optional[int] = 32
    cross_fade_duration: Optional[float] = 0.15

class TTSGenerateRequest(BaseModel):
    audio_file_id: str
    text: str
    ref_text: Optional[str] = ""  # Empty string triggers auto-transcription like Gradio
    settings: Optional[TTSAdvancedSettings] = None  # Advanced settings with validation

class TTSGenerateResponse(BaseModel):
    audio_file_id: str
    ref_text: str
    message: str
    seed_used: int

class PermanentVoiceUploadResponse(BaseModel):
    voice_name: str
    message: str
    duration: Optional[float] = None

class PermanentTTSRequest(BaseModel):
    voice_name: str  # Use permanent voice by name
    text: str
    ref_text: Optional[str] = ""  # Empty string triggers auto-transcription
    settings: Optional[TTSAdvancedSettings] = None

class VoiceCloningRequest(BaseModel):
    text: str
    ref_text: Optional[str] = ""  # Empty string triggers auto-transcription
    settings: Optional[TTSAdvancedSettings] = None

# Initialize models
def load_f5tts():
    """Load F5-TTS model with default configuration."""
    print("Loading F5-TTS model...")
    ckpt_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[0]))
    F5TTS_model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)

def cleanup_file(file_path: str):
    """Background task to clean up temporary files."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up: {file_path}")
    except Exception as e:
        print(f"Error cleaning up {file_path}: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global vocoder, F5TTS_ema_model
    try:
        print("Initializing F5-TTS API server...")
        vocoder = load_vocoder()
        F5TTS_ema_model = load_f5tts()
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "F5-TTS API Server",
        "version": "1.0.0",
        "endpoints": {
            "/upload-audio": "POST - Upload temporary reference audio file",
            "/upload-permanent-voice": "POST - Upload permanent reference voice with custom name",
            "/tts-generate": "POST - Generate TTS using temporary uploaded reference",
            "/tts-permanent": "POST - Generate TTS using permanent reference voice",
            "/voice-cloning": "POST - Upload + generate + delete in one call",
            "/list-voices": "GET - List all permanent reference voices",
            "/docs": "GET - Swagger API documentation",
        },
        "model": DEFAULT_TTS_MODEL,
        "default_settings": DEFAULT_INFERENCE_SETTINGS,
        "advanced_settings": {
            "randomize_seed": "bool - Use random seed for each generation",
            "seed": "int (0-2147483647) - Specific seed for reproducible results",
            "remove_silence": "bool - Auto-detect and crop long silences", 
            "speed": "float (0.3-2.0) - Audio playback speed",
            "nfe_step": "int (4-64) - Number of denoising steps",
            "cross_fade_duration": "float (0.0-1.0) - Cross-fade duration in seconds"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": F5TTS_ema_model is not None,
        "vocoder_loaded": vocoder is not None,
    }

@app.post("/upload-audio", response_model=AudioUploadResponse)
async def upload_audio(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Reference audio file (temporary use - WAV, MP3, etc.)")
):
    """
    Upload temporary reference audio file for TTS generation.
    
    This is for temporary use with /tts-generate endpoint.
    For permanent voices, use /upload-permanent-voice instead.
    Maximum recommended length is 12 seconds for optimal results.
    """
    if not audio_file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    
    # Create temporary file
    temp_dir = tempfile.gettempdir()
    file_extension = Path(audio_file.filename).suffix if audio_file.filename else ".wav"
    temp_file_path = os.path.join(temp_dir, f"f5tts_ref_{file_id}{file_extension}")
    
    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        # Store file mapping
        uploaded_files[file_id] = temp_file_path
        
        # Get audio duration for info
        import librosa
        try:
            duration = librosa.get_duration(filename=temp_file_path)
        except:
            duration = None
        
        # Note: Files will be cleaned up when server restarts or manually deleted
        
        return AudioUploadResponse(
            file_id=file_id,
            message="Audio file uploaded successfully",
            duration=duration
        )
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error processing audio file: {str(e)}")

@app.post("/tts-permanent", response_model=TTSGenerateResponse)
async def tts_permanent(
    request: PermanentTTSRequest
):
    """
    Generate TTS using a permanent reference voice by name.
    
    - Uses permanent voices uploaded via /upload-permanent-voice
    - Reference voices are stored permanently with readable names
    - Same advanced settings as regular TTS generation
    """
    if not F5TTS_ema_model or not vocoder:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Find voice file
    voice_files = [f for f in os.listdir(REFERENCE_VOICES_DIR) 
                   if f.startswith(request.voice_name + ".")]
    
    if not voice_files:
        raise HTTPException(
            status_code=404, 
            detail=f"Voice '{request.voice_name}' not found. Use /list-voices to see available voices."
        )
    
    voice_file_path = os.path.join(REFERENCE_VOICES_DIR, voice_files[0])
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text to generate cannot be empty")
    
    # Use same settings logic as temporary TTS
    final_settings = DEFAULT_INFERENCE_SETTINGS.copy()
    if request.settings:
        user_settings = request.settings.model_dump(exclude_unset=True)
        final_settings.update(user_settings)
    
    # Validate settings
    if final_settings["speed"] < 0.3 or final_settings["speed"] > 2.0:
        raise HTTPException(status_code=400, detail="Speed must be between 0.3 and 2.0")
    if final_settings["nfe_step"] < 4 or final_settings["nfe_step"] > 64:
        raise HTTPException(status_code=400, detail="NFE steps must be between 4 and 64")
    if final_settings["cross_fade_duration"] < 0.0 or final_settings["cross_fade_duration"] > 1.0:
        raise HTTPException(status_code=400, detail="Cross-fade duration must be between 0.0 and 1.0")
    
    # Handle seed logic
    if final_settings["randomize_seed"]:
        final_settings["seed"] = np.random.randint(0, 2**31 - 1)
    elif final_settings["seed"] < 0 or final_settings["seed"] > 2**31 - 1:
        final_settings["seed"] = np.random.randint(0, 2**31 - 1)
    
    torch.manual_seed(final_settings["seed"])
    
    try:
        # Preprocess reference audio
        ref_text_input = request.ref_text if request.ref_text and request.ref_text.strip() else ""
        ref_audio, ref_text = preprocess_ref_audio_text(
            voice_file_path, 
            ref_text_input,
            show_info=print
        )
        
        # Generate TTS audio
        final_wave, final_sample_rate, combined_spectrogram = infer_process(
            ref_audio,
            ref_text,
            request.text,
            F5TTS_ema_model,
            vocoder,
            cross_fade_duration=final_settings["cross_fade_duration"],
            nfe_step=final_settings["nfe_step"],
            speed=final_settings["speed"],
            show_info=print,
        )
        
        # Remove silence if requested
        if final_settings["remove_silence"]:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            try:
                sf.write(temp_path, final_wave, final_sample_rate)
                remove_silence_for_generated_wav(temp_path)
                final_wave, _ = torch.load(temp_path)
                final_wave = final_wave.squeeze().cpu().numpy()
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # Save generated audio
        output_file_id = str(uuid.uuid4())
        output_file_path = os.path.join(tempfile.gettempdir(), f"f5tts_output_{output_file_id}.wav")
        sf.write(output_file_path, final_wave, final_sample_rate)
        
        # Store output file mapping
        uploaded_files[output_file_id] = output_file_path
        
        return TTSGenerateResponse(
            audio_file_id=output_file_id,
            ref_text=ref_text,
            message=f"TTS generated using voice '{request.voice_name}'",
            seed_used=final_settings["seed"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating TTS: {str(e)}")

@app.post("/upload-permanent-voice", response_model=PermanentVoiceUploadResponse)
async def upload_permanent_voice(
    voice_name: str = Form(..., description="Custom name for the voice (e.g., 'adam', 'sarah', 'narrator')"),
    audio_file: UploadFile = File(..., description="Reference audio file (WAV, MP3, etc.)")
):
    """
    Upload a permanent reference voice with a custom readable name.
    
    The voice will be stored permanently and can be used with /tts-permanent endpoint.
    Voice names should be readable like 'adam', 'sarah', 'narrator', etc.
    """
    if not audio_file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    # Clean voice name (remove special chars, lowercase)
    voice_name = re.sub(r'[^a-zA-Z0-9_-]', '', voice_name.lower())
    if not voice_name:
        raise HTTPException(status_code=400, detail="Voice name must contain alphanumeric characters")
    
    # Get file extension
    file_extension = Path(audio_file.filename).suffix if audio_file.filename else ".wav"
    voice_file_path = os.path.join(REFERENCE_VOICES_DIR, f"{voice_name}{file_extension}")
    
    try:
        # Save permanent voice file
        with open(voice_file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        # Get audio duration for info
        import librosa
        try:
            duration = librosa.get_duration(filename=voice_file_path)
        except:
            duration = None
        
        return PermanentVoiceUploadResponse(
            voice_name=voice_name,
            message=f"Permanent voice '{voice_name}' uploaded successfully",
            duration=duration
        )
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(voice_file_path):
            os.remove(voice_file_path)
        raise HTTPException(status_code=500, detail=f"Error saving permanent voice: {str(e)}")
async def tts_generate(
    background_tasks: BackgroundTasks,
    request: TTSGenerateRequest
):
    """
    Generate TTS audio using uploaded reference audio and input text.
    
    - Uses F5-TTS v1 model with exact Gradio default settings
    - Automatically transcribes reference audio (just like Gradio web interface)
    - Leave ref_text empty for auto-transcription, or provide custom reference text
    - All Gradio advanced settings available: randomize_seed, seed, remove_silence, speed, nfe_step, cross_fade_duration
    - Returns generated audio file that can be downloaded
    """
    if not F5TTS_ema_model or not vocoder:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Check if audio file exists
    if request.audio_file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Audio file not found. Please upload audio first.")
    
    ref_audio_path = uploaded_files[request.audio_file_id]
    if not os.path.exists(ref_audio_path):
        raise HTTPException(status_code=404, detail="Reference audio file has expired. Please re-upload.")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text to generate cannot be empty")
    
    # Merge settings with defaults and validate
    final_settings = DEFAULT_INFERENCE_SETTINGS.copy()
    if request.settings:
        # Convert Pydantic model to dict and merge
        user_settings = request.settings.model_dump(exclude_unset=True)
        final_settings.update(user_settings)
    
    # Validate and adjust settings
    if final_settings["speed"] < 0.3 or final_settings["speed"] > 2.0:
        raise HTTPException(status_code=400, detail="Speed must be between 0.3 and 2.0")
    if final_settings["nfe_step"] < 4 or final_settings["nfe_step"] > 64:
        raise HTTPException(status_code=400, detail="NFE steps must be between 4 and 64")
    if final_settings["cross_fade_duration"] < 0.0 or final_settings["cross_fade_duration"] > 1.0:
        raise HTTPException(status_code=400, detail="Cross-fade duration must be between 0.0 and 1.0")
    
    # Handle seed logic like Gradio
    if final_settings["randomize_seed"]:
        final_settings["seed"] = np.random.randint(0, 2**31 - 1)
    elif final_settings["seed"] < 0 or final_settings["seed"] > 2**31 - 1:
        final_settings["seed"] = np.random.randint(0, 2**31 - 1)
    
    torch.manual_seed(final_settings["seed"])
    
    try:
        # Preprocess reference audio and get/transcribe reference text
        # If ref_text is empty or None, auto-transcribe like Gradio
        ref_text_input = request.ref_text if request.ref_text and request.ref_text.strip() else ""
        ref_audio, ref_text = preprocess_ref_audio_text(
            ref_audio_path, 
            ref_text_input,  # Empty string triggers auto-transcription
            show_info=print
        )
        
        # Generate TTS audio
        final_wave, final_sample_rate, combined_spectrogram = infer_process(
            ref_audio,
            ref_text,
            request.text,
            F5TTS_ema_model,
            vocoder,
            cross_fade_duration=final_settings["cross_fade_duration"],
            nfe_step=final_settings["nfe_step"],
            speed=final_settings["speed"],
            show_info=print,
        )
        
        # Remove silence if requested
        if final_settings["remove_silence"]:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            try:
                sf.write(temp_path, final_wave, final_sample_rate)
                remove_silence_for_generated_wav(temp_path)
                final_wave, _ = torch.load(temp_path)
                final_wave = final_wave.squeeze().cpu().numpy()
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # Save generated audio to temporary file
        output_file_id = str(uuid.uuid4())
        output_file_path = os.path.join(tempfile.gettempdir(), f"f5tts_output_{output_file_id}.wav")
        sf.write(output_file_path, final_wave, final_sample_rate)
        
        # Store output file mapping
        uploaded_files[output_file_id] = output_file_path
        
        # Note: Files will be cleaned up when server restarts or manually deleted
        
        return TTSGenerateResponse(
            audio_file_id=output_file_id,
            ref_text=ref_text,
            message="TTS audio generated successfully",
            seed_used=final_settings["seed"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating TTS audio: {str(e)}")

@app.get("/list-voices")
async def list_voices():
    """
    List all available permanent reference voices.
    """
    try:
        voice_files = os.listdir(REFERENCE_VOICES_DIR)
        voices = []
        
        for file in voice_files:
            if file.startswith('.'):
                continue
                
            voice_name = os.path.splitext(file)[0]
            file_path = os.path.join(REFERENCE_VOICES_DIR, file)
            
            try:
                import librosa
                duration = librosa.get_duration(filename=file_path)
            except:
                duration = None
            
            voices.append({
                "name": voice_name,
                "filename": file,
                "duration": duration
            })
        
        return {
            "voices": voices,
            "total_voices": len(voices),
            "storage_path": REFERENCE_VOICES_DIR
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing voices: {str(e)}")

@app.get("/download-audio/{file_id}")
async def download_audio(file_id: str):
    """
    Download generated audio file by file ID.
    """
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    file_path = uploaded_files[file_id]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file has expired")
    
    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=f"f5tts_audio_{file_id}.wav"
    )

@app.post("/voice-cloning", response_model=TTSGenerateResponse)
async def voice_cloning(
    text: str = Form(..., description="Text to generate"),
    ref_text: str = Form("", description="Reference text (leave empty for auto-transcription)"),
    audio_file: UploadFile = File(..., description="Reference audio file (temporary)"),
    randomize_seed: bool = Form(True, description="Use random seed"),
    seed: int = Form(0, description="Specific seed (used if randomize_seed=false)"),
    remove_silence: bool = Form(False, description="Remove silences"),
    speed: float = Form(1.0, description="Speed (0.3-2.0)"),
    nfe_step: int = Form(32, description="NFE steps (4-64)"),
    cross_fade_duration: float = Form(0.15, description="Cross-fade duration (0.0-1.0)")
):
    """
    Voice cloning: Upload audio + generate TTS + delete reference in one call.
    
    This is perfect for one-time voice cloning where you don't want to store the reference permanently.
    The reference audio is processed, used for generation, and immediately deleted.
    """
    if not F5TTS_ema_model or not vocoder:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not audio_file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text to generate cannot be empty")
    
    # Create temporary reference file
    temp_ref_id = str(uuid.uuid4())
    file_extension = Path(audio_file.filename).suffix if audio_file.filename else ".wav"
    temp_ref_path = os.path.join(tempfile.gettempdir(), f"f5tts_clone_{temp_ref_id}{file_extension}")
    
    try:
        # Save temporary reference audio
        with open(temp_ref_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        # Prepare settings
        settings = {
            "randomize_seed": randomize_seed,
            "seed": seed,
            "remove_silence": remove_silence,
            "speed": speed,
            "nfe_step": nfe_step,
            "cross_fade_duration": cross_fade_duration
        }
        
        # Validate settings
        if settings["speed"] < 0.3 or settings["speed"] > 2.0:
            raise HTTPException(status_code=400, detail="Speed must be between 0.3 and 2.0")
        if settings["nfe_step"] < 4 or settings["nfe_step"] > 64:
            raise HTTPException(status_code=400, detail="NFE steps must be between 4 and 64")
        if settings["cross_fade_duration"] < 0.0 or settings["cross_fade_duration"] > 1.0:
            raise HTTPException(status_code=400, detail="Cross-fade duration must be between 0.0 and 1.0")
        
        # Handle seed
        if settings["randomize_seed"]:
            settings["seed"] = np.random.randint(0, 2**31 - 1)
        elif settings["seed"] < 0 or settings["seed"] > 2**31 - 1:
            settings["seed"] = np.random.randint(0, 2**31 - 1)
        
        torch.manual_seed(settings["seed"])
        
        # Process reference audio
        ref_text_input = ref_text if ref_text and ref_text.strip() else ""
        ref_audio, processed_ref_text = preprocess_ref_audio_text(
            temp_ref_path,
            ref_text_input,
            show_info=print
        )
        
        # Generate TTS
        final_wave, final_sample_rate, combined_spectrogram = infer_process(
            ref_audio,
            processed_ref_text,
            text,
            F5TTS_ema_model,
            vocoder,
            cross_fade_duration=settings["cross_fade_duration"],
            nfe_step=settings["nfe_step"],
            speed=settings["speed"],
            show_info=print,
        )
        
        # Remove silence if requested
        if settings["remove_silence"]:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                silence_temp_path = f.name
            try:
                sf.write(silence_temp_path, final_wave, final_sample_rate)
                remove_silence_for_generated_wav(silence_temp_path)
                final_wave, _ = torch.load(silence_temp_path)
                final_wave = final_wave.squeeze().cpu().numpy()
            finally:
                if os.path.exists(silence_temp_path):
                    os.remove(silence_temp_path)
        
        # Save generated audio
        output_file_id = str(uuid.uuid4())
        output_file_path = os.path.join(tempfile.gettempdir(), f"f5tts_output_{output_file_id}.wav")
        sf.write(output_file_path, final_wave, final_sample_rate)
        
        # Store output file mapping
        uploaded_files[output_file_id] = output_file_path
        
        return TTSGenerateResponse(
            audio_file_id=output_file_id,
            ref_text=processed_ref_text,
            message="Voice cloning completed successfully",
            seed_used=settings["seed"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in voice cloning: {str(e)}")
    finally:
        # Always delete temporary reference file
        if os.path.exists(temp_ref_path):
            os.remove(temp_ref_path)
async def download_audio(file_id: str):
    """
    Download generated audio file by file ID.
    """
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    file_path = uploaded_files[file_id]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file has expired")
    
    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=f"f5tts_audio_{file_id}.wav"
    )

@app.delete("/audio/{file_id}")
async def delete_audio(file_id: str):
    """
    Delete uploaded/generated audio file.
    """
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    file_path = uploaded_files[file_id]
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        del uploaded_files[file_id]
        return {"message": "Audio file deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting audio file: {str(e)}")

@app.get("/list-audio")
async def list_audio():
    """
    List all uploaded/generated audio files.
    """
    return {
        "files": [
            {
                "file_id": file_id,
                "exists": os.path.exists(file_path)
            }
            for file_id, file_path in uploaded_files.items()
        ]
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="F5-TTS FastAPI Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Run the server
    uvicorn.run(
        "f5_tts_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )