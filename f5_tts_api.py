#!/usr/bin/env python3
"""
F5-TTS FastAPI Server - Simplified Version
Provides essential REST API endpoints for F5-TTS model inference.
"""

import os
import json
import uuid
import shutil
import tempfile
import threading
import queue
import secrets
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import soundfile as sf
import aiofiles
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Request, Depends, Header
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

# API Security Configuration
API_SECRET_KEY = "speechora_f5tts_api_key_2025_secure_xyz789"

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

# Models and configuration
DEFAULT_TTS_MODEL = "F5-TTS_v1"
DEFAULT_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]

DEFAULT_INFERENCE_SETTINGS = {
    "randomize_seed": True,
    "seed": 0,
    "remove_silence": False,
    "speed": 1.0,
    "nfe_step": 45,
    "cross_fade_duration": 0.15,
}

# Job Queue Models
class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class JobType(str, Enum):
    VOICE_GENERATE = "voice_generate"
    VOICE_CLONE = "voice_clone"

class Job:
    def __init__(self, job_id: str, job_type: JobType, parameters: Dict[str, Any], priority: int = 0):
        self.job_id = job_id
        self.job_type = job_type
        self.parameters = parameters
        self.priority = priority
        self.status = JobStatus.QUEUED
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.progress: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_type": self.job_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
        }

# Job Queue Manager
class JobQueueManager:
    def __init__(self, max_workers: int = 1):
        self.jobs: Dict[str, Job] = {}
        self.job_queue = queue.PriorityQueue()
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.is_running = True
        self.worker_thread = None
        self.current_job_id: Optional[str] = None
        self._start_worker()
    
    def _start_worker(self):
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def _worker(self):
        while self.is_running:
            try:
                priority, job_id = self.job_queue.get(timeout=1.0)
                if job_id in self.jobs:
                    job = self.jobs[job_id]
                    self.current_job_id = job_id
                    
                    try:
                        job.status = JobStatus.PROCESSING
                        job.started_at = datetime.now()
                        job.progress = 10
                        
                        if job.job_type == JobType.VOICE_GENERATE:
                            result = self._process_voice_generate(job)
                        elif job.job_type == JobType.VOICE_CLONE:
                            result = self._process_voice_clone(job)
                        else:
                            raise ValueError(f"Unknown job type: {job.job_type}")
                        
                        job.status = JobStatus.COMPLETED
                        job.completed_at = datetime.now()
                        job.result = result
                        job.progress = 100
                        
                    except Exception as e:
                        job.status = JobStatus.FAILED
                        job.completed_at = datetime.now()
                        job.error = str(e)
                        job.progress = 0
                    
                    finally:
                        self.current_job_id = None
                        self.job_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")
                continue
    
    def submit_job(self, job_type: JobType, parameters: Dict[str, Any], priority: int = 0) -> str:
        job_id = str(uuid.uuid4())
        job = Job(job_id, job_type, parameters, priority)
        self.jobs[job_id] = job
        self.job_queue.put((-priority, job_id))
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        if job_id in self.jobs:
            return self.jobs[job_id].to_dict()
        return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        job_counts = {status.value: 0 for status in JobStatus}
        for job in self.jobs.values():
            job_counts[job.status.value] += 1
        
        return {
            "queue_size": self.job_queue.qsize(),
            "total_jobs": len(self.jobs),
            "job_counts": job_counts,
            "current_job": self.current_job_id,
        }
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        jobs_to_remove = []
        
        for job_id, job in self.jobs.items():
            if (job.status in [JobStatus.COMPLETED, JobStatus.FAILED] and 
                job.completed_at and job.completed_at < cutoff_time):
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.jobs[job_id]
        
        return len(jobs_to_remove)
    
    def _process_voice_generate(self, job: Job) -> Dict[str, Any]:
        params = job.parameters
        job.progress = 30
        
        voice_name = params["voice_name"]
        voice_path = find_voice_file(voice_name)
        
        job.progress = 50
        inference_settings = params.get("settings", DEFAULT_INFERENCE_SETTINGS.copy())
        
        if inference_settings["randomize_seed"]:
            inference_settings["seed"] = np.random.randint(0, 2**31 - 1)
        
        torch.manual_seed(inference_settings["seed"])
        
        # Load reference text if exists
        ref_text_path = os.path.join(REFERENCE_VOICES_DIR, f"{voice_name}_ref.txt")
        ref_text_input = params.get("ref_text", "")
        if not ref_text_input and os.path.exists(ref_text_path):
            with open(ref_text_path, "r", encoding="utf-8") as f:
                ref_text_input = f.read().strip()
        
        ref_audio, ref_text = preprocess_ref_audio_text(voice_path, ref_text_input, show_info=print)
        
        job.progress = 70
        final_wave, final_sample_rate, _ = infer_process(
            ref_audio, ref_text, params["text"], F5TTS_ema_model, vocoder,
            cross_fade_duration=inference_settings["cross_fade_duration"],
            nfe_step=inference_settings["nfe_step"],
            speed=inference_settings["speed"],
            show_info=print,
        )
        
        if inference_settings["remove_silence"]:
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
        
        # Save to permanent location (generated from permanent voice)
        output_file_id = str(uuid.uuid4())
        output_file_path = os.path.join(GENERATED_AUDIO_DIR, f"generated_{voice_name}_{output_file_id}.wav")
        sf.write(output_file_path, final_wave, final_sample_rate)
        
        register_file(output_file_id, output_file_path)
        job.progress = 90
        
        return {
            "output_file_id": output_file_id,
            "voice_name": voice_name,
            "ref_text": ref_text,
            "gen_text": params["text"],
            "seed": inference_settings["seed"]
        }
    
    def _process_voice_clone(self, job: Job) -> Dict[str, Any]:
        params = job.parameters
        job.progress = 30
        
        temp_ref_path = params["temp_ref_path"]
        
        job.progress = 50
        inference_settings = params.get("settings", DEFAULT_INFERENCE_SETTINGS.copy())
        
        if inference_settings["randomize_seed"]:
            inference_settings["seed"] = np.random.randint(0, 2**31 - 1)
        
        torch.manual_seed(inference_settings["seed"])
        
        ref_audio, ref_text = preprocess_ref_audio_text(temp_ref_path, params.get("ref_text", ""), show_info=print)
        
        job.progress = 70
        final_wave, final_sample_rate, _ = infer_process(
            ref_audio, ref_text, params["gen_text"], F5TTS_ema_model, vocoder,
            cross_fade_duration=inference_settings["cross_fade_duration"],
            nfe_step=inference_settings["nfe_step"],
            speed=inference_settings["speed"],
            show_info=print,
        )
        
        if inference_settings["remove_silence"]:
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
        
        # Save to temporary location
        output_file_id = str(uuid.uuid4())
        output_file_path = os.path.join(TEMP_AUDIO_DIR, f"f5tts_clone_output_{output_file_id}.wav")
        sf.write(output_file_path, final_wave, final_sample_rate)
        
        register_file(output_file_id, output_file_path)
        job.progress = 90
        
        # Clean up temp input file
        if os.path.exists(temp_ref_path):
            os.remove(temp_ref_path)
        
        return {
            "output_file_id": output_file_id,
            "ref_text": ref_text,
            "gen_text": params["gen_text"],
            "seed": inference_settings["seed"]
        }

async def verify_api_key(x_api_key: str = Header(..., description="API key for authentication")):
    if x_api_key != API_SECRET_KEY:
        raise HTTPException(
            status_code=401, 
            detail="Invalid API key. Contact admin for access.",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    return True

app = FastAPI(
    title="F5-TTS API - Simplified",
    description="Simplified REST API for F5-TTS Text-to-Speech model",
    version="2.0.0",
    docs_url=None,  # Disable public docs, use admin-protected endpoint
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
vocoder = None
F5TTS_ema_model = None
uploaded_files: Dict[str, str] = {}
job_queue_manager = None

# Permanent storage for reference voices
REFERENCE_VOICES_DIR = "reference_voices"
GENERATED_AUDIO_DIR = "generated_audio"
TEMP_AUDIO_DIR = "temp_audio"
PERSISTENCE_FILE = "uploaded_files.json"

def load_uploaded_files():
    """Load uploaded files mapping from persistence file."""
    global uploaded_files
    if os.path.exists(PERSISTENCE_FILE):
        try:
            with open(PERSISTENCE_FILE, 'r') as f:
                data = json.load(f)
                # Verify files still exist
                valid_files = {}
                for file_id, file_path in data.items():
                    if os.path.exists(file_path):
                        valid_files[file_id] = file_path
                    else:
                        print(f"Warning: File {file_path} no longer exists, removing from mapping")
                uploaded_files = valid_files
                print(f"Loaded {len(uploaded_files)} persistent audio files")
        except Exception as e:
            print(f"Error loading uploaded files: {e}")
            uploaded_files = {}
    else:
        uploaded_files = {}

def save_uploaded_files():
    """Save uploaded files mapping to persistence file."""
    try:
        with open(PERSISTENCE_FILE, 'w') as f:
            json.dump(uploaded_files, f, indent=2)
    except Exception as e:
        print(f"Error saving uploaded files: {e}")

def register_file(file_id: str, file_path: str):
    """Register a new file and persist the mapping."""
    uploaded_files[file_id] = file_path
    save_uploaded_files()

# Create all necessary directories
os.makedirs(REFERENCE_VOICES_DIR, exist_ok=True)
os.makedirs(GENERATED_AUDIO_DIR, exist_ok=True)
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

def find_voice_file(voice_name: str) -> str:
    supported_extensions = ['.wav', '.mp3', '.MP3', '.flac', '.ogg', '.m4a']
    for ext in supported_extensions:
        voice_path = os.path.join(REFERENCE_VOICES_DIR, f"{voice_name}{ext}")
        if os.path.exists(voice_path):
            return voice_path
    raise FileNotFoundError(f"Permanent voice '{voice_name}' not found")

# Pydantic models
class TTSSettings(BaseModel):
    randomize_seed: Optional[bool] = True
    seed: Optional[int] = 0
    remove_silence: Optional[bool] = False
    speed: Optional[float] = 1.0
    nfe_step: Optional[int] = 32
    cross_fade_duration: Optional[float] = 0.15

class VoiceGenerateRequest(BaseModel):
    voice_name: str
    text: str
    ref_text: Optional[str] = ""
    settings: Optional[TTSSettings] = None

class VoiceResponse(BaseModel):
    audio_file_id: str
    ref_text: str
    message: str
    seed_used: int

class VoiceUploadResponse(BaseModel):
    voice_name: str
    message: str
    duration: Optional[float] = None

# Load models
def load_f5tts():
    print("Loading F5-TTS model...")
    ckpt_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[0]))
    F5TTS_model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)

@app.on_event("startup")
async def startup_event():
    global vocoder, F5TTS_ema_model, job_queue_manager
    try:
        print("Initializing F5-TTS API server...")
        vocoder = load_vocoder()
        F5TTS_ema_model = load_f5tts()
        job_queue_manager = JobQueueManager(max_workers=1)
        print("✅ Models and job queue loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    global job_queue_manager
    if job_queue_manager:
        job_queue_manager.is_running = False

# 1. Health endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": F5TTS_ema_model is not None,
        "vocoder_loaded": vocoder is not None,
    }

@app.get("/docs548", include_in_schema=False)
async def get_docs():
    """API Documentation (Swagger UI)."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="F5-TTS API Documentation",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png"
    )

@app.get("/openapi.json", include_in_schema=False)
async def get_openapi():
    """OpenAPI schema."""
    from fastapi.openapi.utils import get_openapi
    return get_openapi(
        title="F5-TTS API - Simplified",
        version="2.0.0",
        description="Simplified REST API for F5-TTS Text-to-Speech model with permanent voice management and temporary voice cloning",
        routes=app.routes,
    )

# 2. Upload audio (permanent voice)
@app.post("/upload-audio", dependencies=[Depends(verify_api_key)])
async def upload_permanent_voice(
    audio_file: UploadFile = File(...),
    voice_name: str = Form(...)
) -> VoiceUploadResponse:
    """Upload audio file as permanent reference voice."""
    try:
        # Validate file type
        if not audio_file.content_type or not any(ct in audio_file.content_type.lower() for ct in ['audio', 'wav', 'mp3', 'ogg', 'flac']):
            raise HTTPException(status_code=400, detail="Invalid audio file format")
        
        # Clean voice name
        clean_voice_name = "".join(c for c in voice_name if c.isalnum() or c in (' ', '-', '_')).strip()
        if not clean_voice_name:
            raise HTTPException(status_code=400, detail="Invalid voice name")
        
        # Get file extension
        file_extension = os.path.splitext(audio_file.filename or "")[1] or ".wav"
        permanent_file_path = os.path.join(REFERENCE_VOICES_DIR, f"{clean_voice_name}{file_extension}")
        
        # Check if voice already exists
        if os.path.exists(permanent_file_path):
            raise HTTPException(status_code=409, detail=f"Voice '{clean_voice_name}' already exists")
        
        # Save to permanent location
        with open(permanent_file_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        # Get duration
        duration = None
        try:
            import librosa
            y, sr = librosa.load(permanent_file_path, sr=None)
            duration = float(len(y)) / sr
        except Exception:
            pass
        
        return VoiceUploadResponse(
            voice_name=clean_voice_name,
            message=f"Voice '{clean_voice_name}' uploaded successfully as permanent reference",
            duration=duration
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# 3. Voice generate (using predefined voices)
@app.post("/voice-generate", dependencies=[Depends(verify_api_key)])
async def voice_generate(request: VoiceGenerateRequest) -> VoiceResponse:
    """Generate TTS audio using permanent reference voice."""
    try:
        # Find the permanent voice file
        try:
            voice_path = find_voice_file(request.voice_name)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        # Prepare inference settings
        inference_settings = DEFAULT_INFERENCE_SETTINGS.copy()
        if request.settings:
            for key, value in request.settings.model_dump().items():
                if key in inference_settings and value is not None:
                    inference_settings[key] = value
        
        # Handle seed
        if inference_settings["randomize_seed"]:
            inference_settings["seed"] = np.random.randint(0, 2**31 - 1)
        
        torch.manual_seed(inference_settings["seed"])
        
        # Load reference text if exists
        ref_text_path = os.path.join(REFERENCE_VOICES_DIR, f"{request.voice_name}_ref.txt")
        ref_text_input = request.ref_text
        if not ref_text_input and os.path.exists(ref_text_path):
            with open(ref_text_path, "r", encoding="utf-8") as f:
                ref_text_input = f.read().strip()
        
        # Process audio
        ref_audio, ref_text = preprocess_ref_audio_text(voice_path, ref_text_input, show_info=print)
        
        # Generate TTS
        final_wave, final_sample_rate, _ = infer_process(
            ref_audio, ref_text, request.text, F5TTS_ema_model, vocoder,
            cross_fade_duration=inference_settings["cross_fade_duration"],
            nfe_step=inference_settings["nfe_step"],
            speed=inference_settings["speed"],
            show_info=print,
        )
        
        # Remove silence if requested
        if inference_settings["remove_silence"]:
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
        
        # Save to permanent location (since it's from permanent voice)
        output_file_id = str(uuid.uuid4())
        output_file_path = os.path.join(GENERATED_AUDIO_DIR, f"generated_{request.voice_name}_{output_file_id}.wav")
        sf.write(output_file_path, final_wave, final_sample_rate)
        
        register_file(output_file_id, output_file_path)
        
        return VoiceResponse(
            audio_file_id=output_file_id,
            ref_text=ref_text,
            message=f"TTS generated successfully using voice '{request.voice_name}'",
            seed_used=inference_settings["seed"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice generation failed: {str(e)}")

# 4. Voice clone (with temporary upload)
@app.post("/voice-clone", dependencies=[Depends(verify_api_key)])
async def voice_clone(
    audio_file: UploadFile = File(...),
    text: str = Form(...),
    ref_text: str = Form(""),
    randomize_seed: bool = Form(True),
    seed: int = Form(0),
    remove_silence: bool = Form(False),
    speed: float = Form(1.0),
    nfe_step: int = Form(32),
    cross_fade_duration: float = Form(0.15)
) -> VoiceResponse:
    """Voice cloning with temporary upload."""
    temp_file_path = None
    try:
        # Validate file type
        if not audio_file.content_type or not any(ct in audio_file.content_type.lower() for ct in ['audio', 'wav', 'mp3', 'ogg', 'flac']):
            raise HTTPException(status_code=400, detail="Invalid audio file format")
        
        # Save temporary file
        file_extension = os.path.splitext(audio_file.filename or "")[1] or ".wav"
        temp_file_path = os.path.join(TEMP_AUDIO_DIR, f"f5tts_clone_{uuid.uuid4()}{file_extension}")
        
        with open(temp_file_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        # Prepare settings
        inference_settings = {
            "randomize_seed": randomize_seed,
            "seed": seed,
            "remove_silence": remove_silence,
            "speed": speed,
            "nfe_step": nfe_step,
            "cross_fade_duration": cross_fade_duration,
        }
        
        # Handle seed
        if inference_settings["randomize_seed"]:
            inference_settings["seed"] = np.random.randint(0, 2**31 - 1)
        
        torch.manual_seed(inference_settings["seed"])
        
        # Process audio
        ref_audio, ref_text_transcribed = preprocess_ref_audio_text(temp_file_path, ref_text, show_info=print)
        
        # Generate TTS
        final_wave, final_sample_rate, _ = infer_process(
            ref_audio, ref_text_transcribed, text, F5TTS_ema_model, vocoder,
            cross_fade_duration=inference_settings["cross_fade_duration"],
            nfe_step=inference_settings["nfe_step"],
            speed=inference_settings["speed"],
            show_info=print,
        )
        
        # Remove silence if requested
        if inference_settings["remove_silence"]:
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
        
        # Save to temporary location
        output_file_id = str(uuid.uuid4())
        output_file_path = os.path.join(TEMP_AUDIO_DIR, f"f5tts_clone_output_{output_file_id}.wav")
        sf.write(output_file_path, final_wave, final_sample_rate)
        
        register_file(output_file_id, output_file_path)
        
        # Clean up input temp file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return VoiceResponse(
            audio_file_id=output_file_id,
            ref_text=ref_text_transcribed,
            message="Voice cloning completed successfully (temporary)",
            seed_used=inference_settings["seed"]
        )
        
    except HTTPException:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise
    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(e)}")

# 5. List voices
@app.get("/list-voices", dependencies=[Depends(verify_api_key)])
async def list_voices():
    """List all permanent reference voices."""
    try:
        voices = []
        if os.path.exists(REFERENCE_VOICES_DIR):
            for file in os.listdir(REFERENCE_VOICES_DIR):
                if file.endswith(('.wav', '.mp3', '.MP3', '.flac', '.ogg', '.m4a')) and not file.startswith('generated_'):
                    file_path = os.path.join(REFERENCE_VOICES_DIR, file)
                    file_size = os.path.getsize(file_path)
                    voice_name = os.path.splitext(file)[0]
                    voices.append({
                        "voice_name": voice_name,
                        "filename": file,
                        "size": f"{file_size / 1024 / 1024:.1f} MB",
                        "path": file_path
                    })
        return voices
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list voices: {str(e)}")

# 5.1. Stream permanent voice file
@app.get("/stream-voice/{voice_name}", dependencies=[Depends(verify_api_key)])
async def stream_permanent_voice(voice_name: str, request: Request):
    """Stream permanent voice file with optimized performance for large files."""
    try:
        # Find the permanent voice file
        voice_path = find_voice_file(voice_name)
        
        # For large voice files, use direct FileResponse for better performance
        file_size = os.path.getsize(voice_path)
        
        if file_size > 10 * 1024 * 1024:  # If file > 10MB, use direct FileResponse
            return FileResponse(
                path=voice_path,
                media_type='audio/wav',
                filename=voice_name,
                headers={
                    "Accept-Ranges": "bytes",
                    "Cache-Control": "public, max-age=3600"  # Cache for 1 hour
                }
            )
        else:
            return await stream_file(voice_path, request)
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stream voice: {str(e)}")

# 6.1. Download all voices
@app.get("/download-all-voices", dependencies=[Depends(verify_api_key)])
async def download_all_voices():
    """Download all permanent reference voices as a ZIP file."""
    import zipfile
    import io
    from fastapi.responses import StreamingResponse
    
    try:
        # Create a ZIP file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            if os.path.exists(REFERENCE_VOICES_DIR):
                for file in os.listdir(REFERENCE_VOICES_DIR):
                    if file.endswith(('.wav', '.mp3', '.MP3', '.flac', '.ogg', '.m4a')) and not file.startswith('generated_'):
                        file_path = os.path.join(REFERENCE_VOICES_DIR, file)
                        zip_file.write(file_path, file)
        
        zip_buffer.seek(0)
        
        # Return the ZIP file as a streaming response
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=all_voices.zip"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create ZIP file: {str(e)}")

# 6. Delete voice
@app.delete("/delete-voice", dependencies=[Depends(verify_api_key)])
async def delete_voice(voice_name: str):
    """Delete a permanent reference voice and its generated files."""
    try:
        # Find and delete the main voice file
        voice_deleted = False
        if os.path.exists(REFERENCE_VOICES_DIR):
            for file in os.listdir(REFERENCE_VOICES_DIR):
                if os.path.splitext(file)[0] == voice_name and file.endswith(('.wav', '.mp3', '.MP3', '.flac', '.ogg', '.m4a')):
                    file_path = os.path.join(REFERENCE_VOICES_DIR, file)
                    os.remove(file_path)
                    voice_deleted = True
                    break
        
        if not voice_deleted:
            raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")
        
        # Delete associated reference text file
        ref_text_path = os.path.join(REFERENCE_VOICES_DIR, f"{voice_name}_ref.txt")
        ref_text_deleted = False
        if os.path.exists(ref_text_path):
            os.remove(ref_text_path)
            ref_text_deleted = True
        
        # Delete all generated files for this voice
        generated_count = 0
        if os.path.exists(GENERATED_AUDIO_DIR):
            for file in os.listdir(GENERATED_AUDIO_DIR):
                if file.startswith(f"generated_{voice_name}_") and file.endswith('.wav'):
                    file_path = os.path.join(GENERATED_AUDIO_DIR, file)
                    # Also remove from uploaded_files mapping
                    file_ids_to_remove = [fid for fid, fpath in uploaded_files.items() if fpath == file_path]
                    for fid in file_ids_to_remove:
                        del uploaded_files[fid]
                    os.remove(file_path)
                    generated_count += 1
        
        # Save updated mapping
        save_uploaded_files()
        
        return {
            "message": f"Voice '{voice_name}' deleted successfully",
            "voice_file_deleted": True,
            "reference_text_deleted": ref_text_deleted,
            "generated_files_deleted": generated_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete voice: {str(e)}")

# 7. Download audio file
@app.get("/download/{file_id}", dependencies=[Depends(verify_api_key)])
async def download_audio_file(file_id: str):
    """Download generated audio file."""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    file_path = uploaded_files[file_id]
    if not os.path.exists(file_path):
        # Clean up dead reference
        del uploaded_files[file_id]
        raise HTTPException(status_code=404, detail="Audio file no longer exists")
    
    return FileResponse(
        path=file_path,
        media_type='application/octet-stream',
        filename=f"f5tts_audio_{file_id}.wav"
    )

# 7.1. Stream audio file (for large files)
@app.get("/stream/{file_id}", dependencies=[Depends(verify_api_key)])
async def stream_audio_file(file_id: str, request: Request):
    """Stream generated audio file with optimized performance for large files."""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    file_path = uploaded_files[file_id]
    if not os.path.exists(file_path):
        # Clean up dead reference
        del uploaded_files[file_id]
        raise HTTPException(status_code=404, detail="Audio file no longer exists")
    
    # For large files, use direct FileResponse with streaming support
    # This is much faster than manual chunking for files > 10MB
    file_size = os.path.getsize(file_path)
    
    if file_size > 10 * 1024 * 1024:  # If file > 10MB, use direct FileResponse
        return FileResponse(
            path=file_path,
            media_type='audio/wav',
            filename=f"f5tts_audio_{file_id}.wav",
            headers={
                "Accept-Ranges": "bytes",
                "Cache-Control": "public, max-age=3600"  # Cache for 1 hour
            }
        )
    else:
        # For smaller files, use custom streaming with range support
        return await stream_file(file_path, request)

async def stream_file(file_path: str, request: Request):
    """Stream a file with range request support for large files."""
    import aiofiles
    from fastapi.responses import StreamingResponse
    
    file_size = os.path.getsize(file_path)
    
    # Handle range requests for large files
    range_header = request.headers.get('range')
    
    if range_header:
        # Parse range header: "bytes=start-end"
        range_match = range_header.replace('bytes=', '').split('-')
        start = int(range_match[0]) if range_match[0] else 0
        end = int(range_match[1]) if range_match[1] else file_size - 1
        
        # Ensure valid range
        start = max(0, start)
        end = min(file_size - 1, end)
        content_length = end - start + 1
        
        async def stream_range():
            async with aiofiles.open(file_path, 'rb') as f:
                await f.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk_size = min(8192, remaining)  # 8KB chunks
                    chunk = await f.read(chunk_size)
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk
        
        headers = {
            'Content-Range': f'bytes {start}-{end}/{file_size}',
            'Accept-Ranges': 'bytes',
            'Content-Length': str(content_length),
            'Content-Type': 'audio/wav'
        }
        
        return StreamingResponse(
            stream_range(),
            status_code=206,  # Partial Content
            headers=headers,
            media_type='audio/wav'
        )
    
    else:
        # Stream entire file
        async def stream_entire():
            async with aiofiles.open(file_path, 'rb') as f:
                while True:
                    chunk = await f.read(8192)  # 8KB chunks
                    if not chunk:
                        break
                    yield chunk
        
        headers = {
            'Content-Length': str(file_size),
            'Accept-Ranges': 'bytes',
            'Content-Type': 'audio/wav'
        }
        
        return StreamingResponse(
            stream_entire(),
            headers=headers,
            media_type='audio/wav'
        )

# 8. Job endpoints

@app.post("/jobs/voice-generate-async", dependencies=[Depends(verify_api_key)])
async def submit_voice_generate_job(request: VoiceGenerateRequest):
    """Submit async job for predefined voice generation."""
    try:
        # Validate voice exists
        find_voice_file(request.voice_name)
        
        parameters = {
            "voice_name": request.voice_name,
            "text": request.text,
            "ref_text": request.ref_text,
            "settings": request.settings.model_dump() if request.settings else DEFAULT_INFERENCE_SETTINGS
        }
        
        job_id = job_queue_manager.submit_job(JobType.VOICE_GENERATE, parameters, priority=0)
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Voice generation job submitted successfully",
            "job_type": "voice_generate"
        }
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Job submission failed: {str(e)}")

@app.post("/jobs/voice-clone-async", dependencies=[Depends(verify_api_key)])
async def submit_voice_clone_job(
    audio_file: UploadFile = File(...),
    gen_text: str = Form(...),
    ref_text: str = Form(""),
    priority: int = Form(0)
):
    """Submit async job for voice cloning."""
    try:
        # Validate file
        if not audio_file.content_type or not any(ct in audio_file.content_type.lower() for ct in ['audio', 'wav', 'mp3', 'ogg', 'flac']):
            raise HTTPException(status_code=400, detail="Invalid audio file format")
        
        # Save temporary file
        file_extension = os.path.splitext(audio_file.filename or "")[1] or ".wav"
        temp_ref_path = os.path.join(TEMP_AUDIO_DIR, f"f5tts_job_{uuid.uuid4()}{file_extension}")
        
        with open(temp_ref_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        parameters = {
            "temp_ref_path": temp_ref_path,
            "gen_text": gen_text,
            "ref_text": ref_text,
            "settings": DEFAULT_INFERENCE_SETTINGS
        }
        
        job_id = job_queue_manager.submit_job(JobType.VOICE_CLONE, parameters, priority=priority)
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Voice cloning job submitted successfully",
            "job_type": "voice_clone"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Job submission failed: {str(e)}")

@app.get("/jobs/{job_id}/status", dependencies=[Depends(verify_api_key)])
async def get_job_status(job_id: str):
    """Get job status."""
    status = job_queue_manager.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status

@app.get("/jobs/queue-status", dependencies=[Depends(verify_api_key)])
async def get_queue_status():
    """Get overall queue status."""
    return job_queue_manager.get_queue_status()

if __name__ == "__main__":
    print("🎵 Starting F5-TTS API Server...")
    print("📁 Loading persistent audio files...")
    load_uploaded_files()
    print("🚀 Server ready!")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)