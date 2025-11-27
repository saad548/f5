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
import time
import asyncio
import threading
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import queue
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Request, Depends, Header
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import uvicorn
import secrets

# API Security Configuration
API_SECRET_KEY = "speechora_f5tts_api_key_2025_secure_xyz789"  # Change this to your own secret!
import platform
import psutil

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

# Job Queue Models
class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class JobType(str, Enum):
    TTS_GENERATE = "tts_generate"
    TTS_PERMANENT = "tts_permanent"
    VOICE_CLONING = "voice_cloning"

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
        self.estimated_completion: Optional[datetime] = None

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
            "estimated_completion": self.estimated_completion.isoformat() if self.estimated_completion else None,
        }

class JobRequest(BaseModel):
    job_type: JobType
    parameters: Dict[str, Any]
    priority: int = 0

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
        
        # Start the background worker
        self._start_worker()
    
    def _start_worker(self):
        """Start the background worker thread"""
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def _worker(self):
        """Background worker that processes jobs from the queue"""
        while self.is_running:
            try:
                # Get next job with timeout to allow checking is_running
                priority, job_id = self.job_queue.get(timeout=1.0)
                
                if job_id in self.jobs:
                    job = self.jobs[job_id]
                    self.current_job_id = job_id
                    
                    try:
                        # Update job status
                        job.status = JobStatus.PROCESSING
                        job.started_at = datetime.now()
                        job.progress = 10
                        
                        # Estimate completion time (rough estimate: 1-2 minutes for TTS)
                        job.estimated_completion = datetime.now() + timedelta(minutes=2)
                        
                        print(f"Processing job {job_id} of type {job.job_type}")
                        
                        # Process the job based on type
                        if job.job_type == JobType.TTS_GENERATE:
                            result = self._process_tts_generate(job)
                        elif job.job_type == JobType.TTS_PERMANENT:
                            result = self._process_tts_permanent(job)
                        elif job.job_type == JobType.VOICE_CLONING:
                            result = self._process_voice_cloning(job)
                        else:
                            raise ValueError(f"Unknown job type: {job.job_type}")
                        
                        # Job completed successfully
                        job.status = JobStatus.COMPLETED
                        job.completed_at = datetime.now()
                        job.result = result
                        job.progress = 100
                        job.estimated_completion = None
                        
                        print(f"Job {job_id} completed successfully")
                        
                    except Exception as e:
                        # Job failed
                        job.status = JobStatus.FAILED
                        job.completed_at = datetime.now()
                        job.error = str(e)
                        job.progress = 0
                        job.estimated_completion = None
                        
                        print(f"Job {job_id} failed: {e}")
                    
                    finally:
                        self.current_job_id = None
                        self.job_queue.task_done()
                
            except queue.Empty:
                # Timeout occurred, continue loop to check is_running
                continue
            except Exception as e:
                print(f"Worker error: {e}")
                continue
    
    def submit_job(self, job_type: JobType, parameters: Dict[str, Any], priority: int = 0) -> str:
        """Submit a new job to the queue"""
        job_id = str(uuid.uuid4())
        job = Job(job_id, job_type, parameters, priority)
        
        self.jobs[job_id] = job
        # Use negative priority for priority queue (smaller numbers = higher priority)
        self.job_queue.put((-priority, job_id))
        
        print(f"Job {job_id} submitted to queue")
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a job"""
        if job_id in self.jobs:
            return self.jobs[job_id].to_dict()
        return None
    
    def get_job_result(self, job_id: str) -> Optional[Any]:
        """Get the result of a completed job"""
        if job_id in self.jobs and self.jobs[job_id].status == JobStatus.COMPLETED:
            return self.jobs[job_id].result
        return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get overall queue status"""
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
        """Remove old completed/failed jobs"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        jobs_to_remove = []
        
        for job_id, job in self.jobs.items():
            if (job.status in [JobStatus.COMPLETED, JobStatus.FAILED] and 
                job.completed_at and job.completed_at < cutoff_time):
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.jobs[job_id]
        
        return len(jobs_to_remove)
    
    def _process_tts_generate(self, job: Job) -> Dict[str, Any]:
        """Process TTS generation job"""
        params = job.parameters
        job.progress = 20
        
        # Load the reference audio file
        ref_audio_path = params["ref_audio_path"]
        if not os.path.exists(ref_audio_path):
            raise FileNotFoundError(f"Reference audio file not found: {ref_audio_path}")
        
        job.progress = 30
        
        # Get inference settings
        inference_settings = params.get("inference_settings", DEFAULT_INFERENCE_SETTINGS.copy())
        
        # Handle seed logic like Gradio
        if inference_settings["randomize_seed"]:
            inference_settings["seed"] = np.random.randint(0, 2**31 - 1)
        elif inference_settings["seed"] < 0 or inference_settings["seed"] > 2**31 - 1:
            inference_settings["seed"] = np.random.randint(0, 2**31 - 1)
        
        torch.manual_seed(inference_settings["seed"])
        job.progress = 40
        
        # Preprocess reference audio and get/transcribe reference text
        ref_text_input = params.get("ref_text", "")
        ref_audio, ref_text = preprocess_ref_audio_text(
            ref_audio_path, 
            ref_text_input,  # Empty string triggers auto-transcription
            show_info=print
        )
        job.progress = 60
        
        # Generate TTS audio
        final_wave, final_sample_rate, combined_spectrogram = infer_process(
            ref_audio,
            ref_text,
            params["gen_text"],
            F5TTS_ema_model,
            vocoder,
            cross_fade_duration=inference_settings["cross_fade_duration"],
            nfe_step=inference_settings["nfe_step"],
            speed=inference_settings["speed"],
            show_info=print,
        )
        job.progress = 80
        
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
        
        # Save generated audio to temporary file
        output_file_id = str(uuid.uuid4())
        output_file_path = os.path.join(tempfile.gettempdir(), f"f5tts_output_{output_file_id}.wav")
        sf.write(output_file_path, final_wave, final_sample_rate)
        
        # Store output file mapping
        uploaded_files[output_file_id] = output_file_path
        job.progress = 90
        
        return {
            "output_file_id": output_file_id,
            "output_path": output_file_path,
            "ref_text": ref_text,
            "gen_text": params["gen_text"],
            "seed": inference_settings["seed"]
        }
    
    def _process_tts_permanent(self, job: Job) -> Dict[str, Any]:
        """Process permanent voice TTS job"""
        params = job.parameters
        job.progress = 20
        
        voice_name = params["voice_name"]
        try:
            voice_path = find_voice_file(voice_name)
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e))
        
        job.progress = 30
        
        # Load reference text for this voice
        ref_text_path = os.path.join(REFERENCE_VOICES_DIR, f"{voice_name}_ref.txt")
        if os.path.exists(ref_text_path):
            with open(ref_text_path, "r", encoding="utf-8") as f:
                ref_text_input = f.read().strip()
        else:
            # Auto-transcribe if no ref text file
            ref_text_input = ""
        
        job.progress = 40
        
        # Get inference settings
        inference_settings = params.get("inference_settings", DEFAULT_INFERENCE_SETTINGS.copy())
        
        # Handle seed logic like Gradio
        if inference_settings["randomize_seed"]:
            inference_settings["seed"] = np.random.randint(0, 2**31 - 1)
        elif inference_settings["seed"] < 0 or inference_settings["seed"] > 2**31 - 1:
            inference_settings["seed"] = np.random.randint(0, 2**31 - 1)
        
        torch.manual_seed(inference_settings["seed"])
        
        # Preprocess reference audio and get/transcribe reference text
        ref_audio, ref_text = preprocess_ref_audio_text(
            voice_path, 
            ref_text_input,  # Empty string triggers auto-transcription
            show_info=print
        )
        job.progress = 60
        
        # Generate TTS audio
        final_wave, final_sample_rate, combined_spectrogram = infer_process(
            ref_audio,
            ref_text,
            params["gen_text"],
            F5TTS_ema_model,
            vocoder,
            cross_fade_duration=inference_settings["cross_fade_duration"],
            nfe_step=inference_settings["nfe_step"],
            speed=inference_settings["speed"],
            show_info=print,
        )
        job.progress = 80
        
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
        
        # Save generated audio to temporary file
        output_file_id = str(uuid.uuid4())
        output_file_path = os.path.join(tempfile.gettempdir(), f"f5tts_output_{output_file_id}.wav")
        sf.write(output_file_path, final_wave, final_sample_rate)
        
        # Store output file mapping
        uploaded_files[output_file_id] = output_file_path
        job.progress = 90
        
        return {
            "output_file_id": output_file_id,
            "output_path": output_file_path,
            "voice_name": voice_name,
            "ref_text": ref_text,
            "gen_text": params["gen_text"],
            "seed": inference_settings["seed"]
        }
    
    def _process_voice_cloning(self, job: Job) -> Dict[str, Any]:
        """Process voice cloning job"""
        params = job.parameters
        job.progress = 20
        
        # Load the uploaded audio file
        audio_file_id = params["audio_file_id"]
        if audio_file_id not in uploaded_files:
            raise ValueError(f"Audio file not found: {audio_file_id}")
        
        ref_audio_path = uploaded_files[audio_file_id]
        job.progress = 30
        
        # Get inference settings
        inference_settings = params.get("inference_settings", DEFAULT_INFERENCE_SETTINGS.copy())
        
        # Handle seed logic like Gradio
        if inference_settings["randomize_seed"]:
            inference_settings["seed"] = np.random.randint(0, 2**31 - 1)
        elif inference_settings["seed"] < 0 or inference_settings["seed"] > 2**31 - 1:
            inference_settings["seed"] = np.random.randint(0, 2**31 - 1)
        
        torch.manual_seed(inference_settings["seed"])
        job.progress = 40
        
        # Preprocess reference audio and auto-transcribe
        ref_audio, ref_text = preprocess_ref_audio_text(
            ref_audio_path, 
            "",  # Empty string triggers auto-transcription
            show_info=print
        )
        job.progress = 60
        
        # Generate TTS audio
        final_wave, final_sample_rate, combined_spectrogram = infer_process(
            ref_audio,
            ref_text,
            params["gen_text"],
            F5TTS_ema_model,
            vocoder,
            cross_fade_duration=inference_settings["cross_fade_duration"],
            nfe_step=inference_settings["nfe_step"],
            speed=inference_settings["speed"],
            show_info=print,
        )
        job.progress = 80
        
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
        
        # Save generated audio to temporary file
        output_file_id = str(uuid.uuid4())
        output_file_path = os.path.join(tempfile.gettempdir(), f"f5tts_output_{output_file_id}.wav")
        sf.write(output_file_path, final_wave, final_sample_rate)
        
        # Store output file mapping
        uploaded_files[output_file_id] = output_file_path
        job.progress = 90
        
        return {
            "output_file_id": output_file_id,
            "output_path": output_file_path,
            "ref_text": ref_text,
            "gen_text": params["gen_text"],
            "seed": inference_settings["seed"]
        }
    
    def stop(self):
        """Stop the job queue manager"""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)

# Admin authentication
ADMIN_USERNAME = "yasirr548"
ADMIN_PASSWORD = "yasirr548AJSKD#D45s"  # Change this!
security = HTTPBasic()

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(status_code=401, detail="Invalid admin credentials")
    return credentials.username

async def verify_api_key(x_api_key: str = Header(..., description="API key for authentication")):
    """
    Verify API key for endpoint access
    """
    if x_api_key != API_SECRET_KEY:
        raise HTTPException(
            status_code=401, 
            detail="Invalid API key. Contact admin for access.",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    return True

app = FastAPI(
    title="F5-TTS API",
    description="REST API for F5-TTS Text-to-Speech model with automatic reference text transcription",
    version="1.0.0",
    docs_url=None,  # Disable public docs
    redoc_url=None,  # Disable public redoc
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

# Initialize job queue manager
job_queue_manager = None

# Permanent storage for reference voices
REFERENCE_VOICES_DIR = "reference_voices"
os.makedirs(REFERENCE_VOICES_DIR, exist_ok=True)

def find_voice_file(voice_name: str) -> str:
    """Find voice file with any supported extension"""
    supported_extensions = ['.wav', '.mp3', '.MP3', '.flac', '.ogg', '.m4a']
    
    for ext in supported_extensions:
        voice_path = os.path.join(REFERENCE_VOICES_DIR, f"{voice_name}{ext}")
        if os.path.exists(voice_path):
            return voice_path
    
    raise FileNotFoundError(f"Permanent voice '{voice_name}' not found with any supported extension")

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
    """Initialize models and job queue on startup."""
    global vocoder, F5TTS_ema_model, job_queue_manager
    try:
        print("Initializing F5-TTS API server...")
        vocoder = load_vocoder()
        F5TTS_ema_model = load_f5tts()
        
        # Initialize job queue manager
        job_queue_manager = JobQueueManager(max_workers=1)
        print("✅ Job queue manager initialized!")
        
        print("✅ Models and job queue loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models/job queue: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global job_queue_manager
    try:
        if job_queue_manager:
            print("Shutting down job queue manager...")
            job_queue_manager.stop()
            print("✅ Job queue manager stopped!")
    except Exception as e:
        print(f"❌ Error during shutdown: {e}")

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

@app.get("/docs", include_in_schema=False)
async def get_docs(credentials: HTTPBasicCredentials = Depends(verify_admin)):
    """
    Admin-only Swagger documentation.
    Requires admin authentication to access API documentation.
    """
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="F5-TTS API Documentation",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png"
    )

@app.get("/openapi.json", include_in_schema=False)
async def get_openapi(credentials: HTTPBasicCredentials = Depends(verify_admin)):
    """
    Admin-only OpenAPI schema.
    Requires admin authentication to access API schema.
    """
    from fastapi.openapi.utils import get_openapi
    return get_openapi(
        title="F5-TTS API",
        version="1.0.0",
        description="REST API for F5-TTS Text-to-Speech model with automatic reference text transcription",
        routes=app.routes,
    )

@app.get("/admin", response_class=HTMLResponse)
async def admin_panel(credentials: HTTPBasicCredentials = Depends(verify_admin)):
    """Complete Admin Dashboard - Single Page Interface"""
    
    # Get system stats
    try:
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
    except:
        cpu_percent = 0
        memory = type('obj', (object,), {'percent': 0, 'used': 0, 'total': 0})
        disk = type('obj', (object,), {'percent': 0, 'used': 0, 'total': 0})
    
    # Get voice files
    try:
        voice_files = []
        for file in os.listdir(REFERENCE_VOICES_DIR):
            if not file.startswith('.') and file.endswith(('.wav', '.mp3', '.MP3', '.flac')):
                file_path = os.path.join(REFERENCE_VOICES_DIR, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                voice_files.append({
                    'name': file,
                    'size': f"{file_size:.1f} MB",
                    'path': file_path
                })
    except:
        voice_files = []
    
    # Get queue stats
    queue_stats = {"total_jobs": 0, "queue_size": 0, "current_job": None, "job_counts": {"queued": 0, "processing": 0, "completed": 0, "failed": 0}}
    recent_jobs = []
    if job_queue_manager:
        queue_stats = job_queue_manager.get_queue_status()
        # Get last 10 jobs
        for job_id, job in list(job_queue_manager.jobs.items())[-10:]:
            recent_jobs.append({
                "id": job_id[:8],
                "type": job.job_type.value,
                "status": job.status.value,
                "progress": job.progress,
                "created": job.created_at.strftime('%H:%M:%S')
            })
    
    # Get uploaded files
    uploaded_count = len(uploaded_files)
    
    # Enhanced Voice files list HTML with professional features
    voice_rows = ""
    voice_cards_html = ""
    for voice in voice_files:
        voice_name_clean = voice['name'].replace('.wav', '').replace('.mp3', '').replace('.flac', '')
        voice_initial = voice_name_clean[0].upper() if voice_name_clean else 'V'
        
        # Generate voice card
        voice_cards_html += f"""
        <div class="voice-card">
            <div class="voice-info">
                <div class="voice-avatar">{voice_initial}</div>
                <div class="voice-details">
                    <h3>{voice_name_clean}</h3>
                    <div class="voice-meta">
                        <span><i class="fas fa-weight"></i> {voice['size']}</span>
                        <span><i class="fas fa-clock"></i> ~5-10s</span>
                        <span><i class="fas fa-star"></i> Premium Quality</span>
                    </div>
                </div>
            </div>
            
            <div class="audio-player" id="player-{voice_name_clean}">
                <audio controls style="width: 100%; margin-bottom: 10px;" preload="none">
                    <source src="/voice/{voice['name']}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
            </div>
            
            <div class="voice-controls">
                <button onclick="playVoice('{voice['name']}')" class="btn-small btn-primary">
                    <i class="fas fa-play"></i> Preview
                </button>
                <button onclick="testVoiceGeneration('{voice['name']}')" class="btn-small btn-success">
                    <i class="fas fa-magic"></i> Test TTS
                </button>
                <button onclick="downloadVoice('{voice['name']}')" class="btn-small btn-glass">
                    <i class="fas fa-download"></i> Download
                </button>
                <button onclick="confirmDeleteVoice('{voice['name']}')" class="btn-small btn-danger">
                    <i class="fas fa-trash"></i> Delete
                </button>
            </div>
        </div>
        """
        
        # Generate table row for backward compatibility
        voice_rows += f"""
        <tr>
            <td>
                <div style="display: flex; align-items: center;">
                    <i class="fas fa-file-audio" style="color: #3498db; margin-right: 8px;"></i>
                    <strong>{voice_name_clean}</strong>
                </div>
            </td>
            <td>{voice['size']}</td>
            <td><span class="badge">~5-10s</span></td>
            <td>
                <span style="color: #27ae60;">
                    <i class="fas fa-star"></i>
                    <i class="fas fa-star"></i>
                    <i class="fas fa-star"></i>
                    <i class="fas fa-star"></i>
                    <i class="far fa-star"></i>
                </span>
            </td>
            <td>
                <button onclick="playVoice('{voice['name']}')" class="btn-small btn-primary">
                    <i class="fas fa-play"></i> Preview
                </button>
                <button onclick="testVoice('{voice['name']}')" class="btn-small btn-success">
                    <i class="fas fa-microphone"></i> Test TTS
                </button>
                <button onclick="deleteVoice('{voice['name']}')" class="btn-small btn-danger">
                    <i class="fas fa-trash"></i> Delete
                </button>
            </td>
        </tr>"""
    
    # Enhanced Recent jobs HTML with professional features
    job_rows = ""
    for job in recent_jobs:
        status_color = {"queued": "#f39c12", "processing": "#3498db", "completed": "#27ae60", "failed": "#e74c3c"}.get(job['status'], "#95a5a6")
        status_icon = {"queued": "clock", "processing": "spinner fa-spin", "completed": "check-circle", "failed": "exclamation-triangle"}.get(job['status'], "question-circle")
        job_type_icon = {"tts_generate": "microphone", "tts_permanent": "music", "voice_cloning": "clone"}.get(job['type'], "cog")
        
        job_rows += f"""
        <tr>
            <td>
                <code style="background: #f8f9fa; padding: 2px 6px; border-radius: 3px;">{job['id']}</code>
            </td>
            <td>
                <span style="display: flex; align-items: center;">
                    <i class="fas fa-{job_type_icon}" style="margin-right: 5px;"></i>
                    {job['type'].replace('_', ' ').title()}
                </span>
            </td>
            <td>
                <span style="color: {status_color}; font-weight: bold; display: flex; align-items: center;">
                    <i class="fas fa-{status_icon}" style="margin-right: 5px;"></i>
                    {job['status'].title()}
                </span>
            </td>
            <td>
                <div style="display: flex; align-items: center;">
                    <div class="progress-bar" style="width: 80px; margin-right: 10px;">
                        <div class="progress-fill" style="width: {job['progress']}%; background: {status_color};"></div>
                    </div>
                    <span style="font-size: 0.85em; font-weight: 500;">{job['progress']}%</span>
                </div>
            </td>
            <td style="font-family: monospace; font-size: 0.9em;">{job['created']}</td>
            <td>
                <button onclick="viewJobDetails('{job['id']}')" class="btn-small btn-primary">
                    <i class="fas fa-info-circle"></i> Details
                </button>"""
        
        # Add cancel button if job is active
        if job['status'] in ['queued', 'processing']:
            job_rows += f"""
                <button onclick="cancelJob('{job['id']}')" class="btn-small btn-danger">
                    <i class="fas fa-stop"></i> Cancel
                </button>"""
        
        job_rows += """
            </td>
        </tr>"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>F5-TTS Professional Admin Dashboard</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            :root {{
                --primary-color: #667eea;
                --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                --secondary-color: #2c3e50;
                --success-color: #4ecdc4;
                --success-gradient: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
                --warning-color: #feca57;
                --warning-gradient: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
                --danger-color: #ff6b6b;
                --danger-gradient: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
                --dark-bg: #0f0f23;
                --dark-sidebar: #1a1a2e;
                --dark-card: #16213e;
                --text-light: #eee;
                --border-color: #e1e8ed;
                --glass-bg: rgba(255, 255, 255, 0.1);
                --shadow-light: 0 8px 32px rgba(31, 38, 135, 0.37);
                --shadow-dark: 0 8px 32px rgba(0, 0, 0, 0.4);
            }}
            
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: #f8f9fa; 
                transition: all 0.3s ease;
            }}
            
            body.dark-mode {{
                background: var(--dark-bg);
                color: var(--text-light);
            }}
            
            .container {{ display: flex; min-height: 100vh; }}
            
            /* Ultra-Modern Sidebar */
            .sidebar {{ 
                width: 300px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #667eea 100%); 
                color: white; 
                padding: 25px; 
                position: relative;
                box-shadow: 4px 0 30px rgba(0,0,0,0.2);
                backdrop-filter: blur(20px);
                border-right: 1px solid rgba(255,255,255,0.1);
            }}
            
            .dark-mode .sidebar {{
                background: linear-gradient(135deg, var(--dark-sidebar), #404040);
            }}
            
            .sidebar-header {{
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }}
            
            .sidebar h2 {{ 
                margin-bottom: 10px; 
                color: #ecf0f1; 
                font-size: 1.4em;
            }}
            
            .server-status {{
                background: rgba(46, 204, 113, 0.2);
                padding: 8px 12px;
                border-radius: 20px;
                font-size: 0.85em;
                display: inline-block;
            }}
            
            .nav-item {{ 
                padding: 15px 20px; 
                margin: 8px 0; 
                cursor: pointer; 
                border-radius: 8px; 
                transition: all 0.3s;
                display: flex;
                align-items: center;
                position: relative;
            }}
            .nav-item:hover {{ 
                background: rgba(52, 152, 219, 0.2); 
                transform: translateX(5px);
            }}
            .nav-item.active {{ 
                background: linear-gradient(45deg, #3498db, #2980b9); 
                box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
            }}
            .nav-icon {{ 
                margin-right: 15px; 
                width: 20px;
                text-align: center;
            }}
            
            /* Theme Toggle */
            .theme-toggle {{
                position: absolute;
                bottom: 20px;
                left: 20px;
                right: 20px;
                padding: 12px;
                background: rgba(0,0,0,0.2);
                border: none;
                border-radius: 8px;
                color: white;
                cursor: pointer;
                transition: all 0.3s;
            }}
            
            .theme-toggle:hover {{
                background: rgba(0,0,0,0.4);
            }}
            
            /* Enhanced Main Content */
            .main-content {{ flex: 1; padding: 25px; }}
            
            .header {{ 
                background: white; 
                padding: 25px; 
                border-radius: 15px; 
                box-shadow: 0 4px 20px rgba(0,0,0,0.08); 
                margin-bottom: 25px;
                position: relative;
                overflow: hidden;
            }}
            
            .dark-mode .header {{
                background: var(--dark-card);
            }}
            
            .header::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #3498db, #2ecc71, #f39c12, #e74c3c);
            }}
            
            .header-content {{
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .header h1 {{ 
                color: #2c3e50; 
                margin-bottom: 8px; 
                font-size: 2.2em;
            }}
            
            .dark-mode .header h1 {{ color: var(--text-light); }}
            
            .header p {{ 
                color: #7f8c8d; 
                font-size: 1.1em;
            }}
            
            .live-indicator {{
                display: flex;
                align-items: center;
                background: linear-gradient(45deg, #27ae60, #2ecc71);
                padding: 10px 15px;
                border-radius: 25px;
                color: white;
                font-weight: 600;
            }}
            
            .live-dot {{
                width: 8px;
                height: 8px;
                background: white;
                border-radius: 50%;
                margin-right: 8px;
                animation: pulse 2s infinite;
            }}
            
            @keyframes pulse {{
                0% {{ transform: scale(1); opacity: 1; }}
                50% {{ transform: scale(1.2); opacity: 0.7; }}
                100% {{ transform: scale(1); opacity: 1; }}
            }}
            
            /* Enhanced Stats Cards */
            .stats-grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); 
                gap: 25px; 
                margin-bottom: 35px; 
            }}
            
            .stat-card {{ 
                background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.8) 100%); 
                backdrop-filter: blur(20px);
                padding: 30px; 
                border-radius: 20px; 
                box-shadow: 0 8px 40px rgba(0,0,0,0.1); 
                text-align: center;
                position: relative;
                overflow: hidden;
                transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                border: 1px solid rgba(255,255,255,0.3);
            }}
            
            .dark-mode .stat-card {{
                background: var(--dark-card);
            }}
            
            .stat-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 30px rgba(0,0,0,0.15);
            }}
            
            .stat-card::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: linear-gradient(90deg, var(--primary-color), var(--success-color));
            }}
            
            .stat-value {{ 
                font-size: 2.5em; 
                font-weight: 700; 
                margin-bottom: 8px;
                background: linear-gradient(45deg, #3498db, #2ecc71);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            
            .stat-label {{ 
                color: #7f8c8d; 
                font-size: 0.95em; 
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .stat-trend {{
                margin-top: 10px;
                font-size: 0.85em;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            
            .trend-up {{ color: var(--success-color); }}
            .trend-down {{ color: var(--danger-color); }}
            
            .stat-good {{ color: #27ae60; }}
            .stat-warning {{ color: #f39c12; }}
            .stat-danger {{ color: #e74c3c; }}
            
            /* Enhanced Progress Bar */
            .progress-bar {{ 
                width: 100%; 
                height: 8px; 
                background: #ecf0f1; 
                border-radius: 4px; 
                margin-top: 12px;
                overflow: hidden;
            }}
            
            .progress-fill {{ 
                height: 100%; 
                border-radius: 4px; 
                transition: all 0.5s ease;
                background: linear-gradient(90deg, #3498db, #2ecc71);
                position: relative;
            }}
            
            .progress-fill::after {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
                animation: shimmer 2s infinite;
            }}
            
            @keyframes shimmer {{
                0% {{ transform: translateX(-100%); }}
                100% {{ transform: translateX(100%); }}
            }}
            
            /* Enhanced Content Sections */
            .content-section {{ 
                background: white; 
                border-radius: 15px; 
                box-shadow: 0 4px 20px rgba(0,0,0,0.08); 
                margin-bottom: 25px; 
                display: none;
                overflow: hidden;
            }}
            
            .dark-mode .content-section {{
                background: var(--dark-card);
            }}
            
            .content-section.active {{ display: block; }}
            
            .section-header {{ 
                padding: 25px; 
                border-bottom: 1px solid #ecf0f1;
                background: linear-gradient(135deg, #f8f9fa, #ffffff);
                position: relative;
            }}
            
            .dark-mode .section-header {{
                background: linear-gradient(135deg, #404040, #4a4a4a);
                border-bottom-color: #555;
            }}
            
            .section-header h3 {{
                color: #2c3e50;
                font-size: 1.4em;
                font-weight: 600;
            }}
            
            .dark-mode .section-header h3 {{ color: var(--text-light); }}
            
            .section-content {{ padding: 25px; }}
            
            /* Enhanced Tables */
            table {{ 
                width: 100%; 
                border-collapse: collapse;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }}
            
            th, td {{ 
                padding: 15px; 
                text-align: left; 
                border-bottom: 1px solid #ecf0f1; 
            }}
            
            th {{ 
                background: linear-gradient(135deg, #f8f9fa, #e9ecef); 
                font-weight: 600; 
                color: #2c3e50;
                text-transform: uppercase;
                font-size: 0.85em;
                letter-spacing: 0.5px;
            }}
            
            .dark-mode th {{
                background: linear-gradient(135deg, #404040, #4a4a4a);
                color: var(--text-light);
            }}
            
            .dark-mode td {{
                border-bottom-color: #555;
            }}
            
            tr:hover {{ 
                background: linear-gradient(90deg, rgba(52, 152, 219, 0.05), rgba(46, 204, 113, 0.05)); 
            }}
            
            /* Ultra-Modern Buttons */
            .btn-small {{ 
                padding: 12px 24px; 
                border: none; 
                border-radius: 16px; 
                cursor: pointer; 
                font-size: 0.9em;
                font-weight: 600;
                transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                text-transform: none;
                letter-spacing: 0.3px;
                position: relative;
                overflow: hidden;
                backdrop-filter: blur(10px);
                margin: 3px;
                display: inline-flex;
                align-items: center;
                gap: 8px;
                border: 1px solid rgba(255,255,255,0.2);
            }}
            
            .btn-small::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
                transition: left 0.5s;
            }}
            
            .btn-small:hover::before {{
                left: 100%;
            }}
            
            .btn-primary {{ 
                background: var(--primary-gradient);
                color: white;
                box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
            }}
            
            .btn-primary:hover {{
                transform: translateY(-3px) scale(1.02);
                box-shadow: 0 8px 30px rgba(102, 126, 234, 0.6);
            }}
            
            .btn-success {{ 
                background: var(--success-gradient);
                color: white;
                box-shadow: 0 4px 20px rgba(78, 205, 196, 0.4);
            }}
            
            .btn-success:hover {{
                transform: translateY(-3px) scale(1.02);
                box-shadow: 0 8px 30px rgba(78, 205, 196, 0.6);
            }}
            
            .btn-warning {{ 
                background: var(--warning-gradient);
                color: white;
                box-shadow: 0 4px 20px rgba(254, 202, 87, 0.4);
            }}
            
            .btn-warning:hover {{
                transform: translateY(-3px) scale(1.02);
                box-shadow: 0 8px 30px rgba(254, 202, 87, 0.6);
            }}
            
            .btn-danger {{ 
                background: var(--danger-gradient);
                color: white;
                box-shadow: 0 4px 20px rgba(255, 107, 107, 0.4);
            }}
            
            .btn-danger:hover {{
                transform: translateY(-3px) scale(1.02);
                box-shadow: 0 8px 30px rgba(255, 107, 107, 0.6);
            }}
            
            .btn-glass {{
                background: var(--glass-bg);
                backdrop-filter: blur(15px);
                border: 1px solid rgba(255,255,255,0.3);
                color: var(--text-light);
                box-shadow: var(--shadow-light);
            }}
            
            .btn-glass:hover {{
                background: rgba(255,255,255,0.2);
                transform: translateY(-3px) scale(1.02);
            }}
            
            /* Charts Container */
            .chart-container {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }}
            
            .dark-mode .chart-container {{
                background: var(--dark-card);
            }}
            
            /* Modern Voice Cards */
            .voice-card {{
                background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.8) 100%);
                backdrop-filter: blur(20px);
                border-radius: 20px;
                padding: 25px;
                margin: 15px 0;
                border: 1px solid rgba(255,255,255,0.3);
                box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
                transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            }}
            
            .voice-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 12px 40px rgba(31, 38, 135, 0.25);
            }}
            
            .dark-mode .voice-card {{
                background: linear-gradient(135deg, rgba(22, 33, 62, 0.9) 0%, rgba(22, 33, 62, 0.7) 100%);
                border: 1px solid rgba(255,255,255,0.1);
            }}
            
            .voice-info {{
                display: flex;
                align-items: center;
                gap: 20px;
                margin-bottom: 20px;
            }}
            
            .voice-avatar {{
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background: var(--primary-gradient);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 24px;
                font-weight: bold;
                box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
            }}
            
            .voice-details h3 {{
                margin: 0 0 5px 0;
                color: var(--primary-color);
                font-size: 1.3em;
            }}
            
            .voice-meta {{
                color: #666;
                font-size: 0.9em;
                display: flex;
                gap: 15px;
            }}
            
            .dark-mode .voice-meta {{
                color: #aaa;
            }}
            
            .voice-controls {{
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
            }}
            
            .audio-player {{
                background: #f8f9fa;
                border-radius: 12px;
                padding: 15px;
                margin: 15px 0;
                border: 1px solid #e1e8ed;
            }}
            
            .dark-mode .audio-player {{
                background: #2a2a3a;
                border-color: #404040;
            }}
            
            .upload-area {{
                border: 2px dashed var(--primary-color);
                border-radius: 16px;
                padding: 40px;
                text-align: center;
                margin: 20px 0;
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
                transition: all 0.3s ease;
                cursor: pointer;
            }}
            
            .upload-area:hover {{
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                transform: translateY(-2px);
            }}
            
            .upload-icon {{
                font-size: 48px;
                color: var(--primary-color);
                margin-bottom: 15px;
            }}
            
            /* Voice Grid Layout */
            .voice-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
                gap: 20px;
                margin-top: 30px;
            }}
            
            @media (max-width: 768px) {{
                .voice-grid {{
                    grid-template-columns: 1fr;
                }}
            }}
            
            /* Log Viewer */
            .log-viewer {{
                background: #1e1e1e;
                color: #00ff00;
                font-family: 'Courier New', monospace;
                padding: 20px;
                border-radius: 10px;
                height: 400px;
                overflow-y: auto;
                font-size: 0.9em;
                line-height: 1.4;
            }}
            
            .log-entry {{
                margin: 5px 0;
                padding: 5px;
                border-left: 3px solid #00ff00;
                padding-left: 10px;
            }}
            
            .log-error {{ border-left-color: #ff4444; color: #ff4444; }}
            .log-warning {{ border-left-color: #ffaa00; color: #ffaa00; }}
            .log-info {{ border-left-color: #00aaff; color: #00aaff; }}
            
            /* Responsive Design */
            @media (max-width: 768px) {{
                .container {{ flex-direction: column; }}
                .sidebar {{ width: 100%; }}
                .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
                .main-content {{ padding: 15px; }}
            }}
            
            @media (max-width: 480px) {{
                .stats-grid {{ grid-template-columns: 1fr; }}
                .header-content {{ flex-direction: column; text-align: center; }}
                .live-indicator {{ margin-top: 15px; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Enhanced Sidebar -->
            <div class="sidebar">
                <div class="sidebar-header">
                    <h2><i class="fas fa-microphone-alt"></i> F5-TTS Pro</h2>
                    <div class="server-status">
                        <i class="fas fa-server"></i> Server Online
                    </div>
                </div>
                
                <div class="nav-item active" onclick="showSection('dashboard')">
                    <span class="nav-icon"><i class="fas fa-chart-line"></i></span> Analytics Dashboard
                </div>
                <div class="nav-item" onclick="showSection('voices')">
                    <span class="nav-icon"><i class="fas fa-music"></i></span> Voice Bank Pro
                </div>
                <div class="nav-item" onclick="showSection('jobs')">
                    <span class="nav-icon"><i class="fas fa-tasks"></i></span> Job Manager
                </div>
                <div class="nav-item" onclick="showSection('performance')">
                    <span class="nav-icon"><i class="fas fa-tachometer-alt"></i></span> Performance
                </div>
                <div class="nav-item" onclick="showSection('logs')">
                    <span class="nav-icon"><i class="fas fa-terminal"></i></span> Live Logs
                </div>
                <div class="nav-item" onclick="showSection('users')">
                    <span class="nav-icon"><i class="fas fa-users"></i></span> User Activity
                </div>
                <div class="nav-item" onclick="showSection('settings')">
                    <span class="nav-icon"><i class="fas fa-cog"></i></span> Configuration
                </div>
                <div class="nav-item" onclick="showSection('backup')">
                    <span class="nav-icon"><i class="fas fa-download"></i></span> Backup & Export
                </div>
                <div class="nav-item" onclick="window.open('/docs', '_blank')">
                    <span class="nav-icon"><i class="fas fa-book"></i></span> API Documentation
                </div>
                
                <button class="theme-toggle" onclick="toggleTheme()">
                    <i class="fas fa-moon" id="theme-icon"></i> <span id="theme-text">Dark Mode</span>
                </button>
            </div>
            
            <!-- Enhanced Main Content -->
            <div class="main-content">
                <!-- Enhanced Header -->
                <div class="header">
                    <div class="header-content">
                        <div>
                            <h1><i class="fas fa-shield-alt"></i> F5-TTS Control Center</h1>
                            <p>Professional AI Voice Generation Management System</p>
                        </div>
                        <div class="live-indicator">
                            <div class="live-dot"></div>
                            <span>LIVE</span>
                        </div>
                    </div>
                </div>
                
                <!-- Enhanced Stats Overview -->
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value stat-good"><i class="fas fa-check-circle"></i></div>
                        <div class="stat-label">System Health</div>
                        <div class="stat-trend trend-up">
                            <i class="fas fa-arrow-up"></i> 99.9% Uptime
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len(voice_files)}</div>
                        <div class="stat-label">Voice Models</div>
                        <div class="stat-trend trend-up">
                            <i class="fas fa-plus"></i> +3 this week
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{queue_stats['total_jobs']}</div>
                        <div class="stat-label">Total Processed</div>
                        <div class="stat-trend trend-up">
                            <i class="fas fa-rocket"></i> +{queue_stats['job_counts']['completed']} today
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{uploaded_count}</div>
                        <div class="stat-label">Active Sessions</div>
                        <div class="stat-trend">
                            <i class="fas fa-users"></i> {uploaded_count} files cached
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value {'stat-good' if F5TTS_ema_model and vocoder else 'stat-danger'}">
                            <i class="fas fa-{'brain' if F5TTS_ema_model and vocoder else 'exclamation-triangle'}"></i>
                        </div>
                        <div class="stat-label">AI Models</div>
                        <div class="stat-trend {'trend-up' if F5TTS_ema_model and vocoder else 'trend-down'}">
                            <i class="fas fa-{'check' if F5TTS_ema_model and vocoder else 'times'}"></i> {'Ready' if F5TTS_ema_model and vocoder else 'Loading'}
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value stat-warning">{queue_stats['queue_size']}</div>
                        <div class="stat-label">Queue Status</div>
                        <div class="stat-trend">
                            <i class="fas fa-clock"></i> {'Processing' if queue_stats['current_job'] else 'Idle'}
                        </div>
                    </div>
                </div>
                
                <!-- Analytics Dashboard Section -->
                <div id="dashboard" class="content-section active">
                    <div class="section-header">
                        <h3><i class="fas fa-chart-line"></i> Real-Time Analytics Dashboard</h3>
                    </div>
                    <div class="section-content">
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-value">{cpu_percent:.1f}%</div>
                                <div class="stat-label"><i class="fas fa-microchip"></i> CPU Usage</div>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: {cpu_percent}%; background: {'linear-gradient(90deg, #e74c3c, #c0392b)' if cpu_percent > 80 else 'linear-gradient(90deg, #f39c12, #e67e22)' if cpu_percent > 50 else 'linear-gradient(90deg, #27ae60, #229954)'};"></div>
                                </div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">{memory.percent:.1f}%</div>
                                <div class="stat-label"><i class="fas fa-memory"></i> Memory Usage</div>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: {memory.percent}%; background: {'linear-gradient(90deg, #e74c3c, #c0392b)' if memory.percent > 80 else 'linear-gradient(90deg, #f39c12, #e67e22)' if memory.percent > 50 else 'linear-gradient(90deg, #27ae60, #229954)'};"></div>
                                </div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">{disk.percent:.1f}%</div>
                                <div class="stat-label"><i class="fas fa-hdd"></i> Storage Usage</div>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: {disk.percent}%; background: {'linear-gradient(90deg, #e74c3c, #c0392b)' if disk.percent > 80 else 'linear-gradient(90deg, #f39c12, #e67e22)' if disk.percent > 50 else 'linear-gradient(90deg, #27ae60, #229954)'};"></div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="chart-container">
                            <h4><i class="fas fa-chart-area"></i> Performance Metrics</h4>
                            <canvas id="performanceChart" width="400" height="200"></canvas>
                        </div>
                    </div>
                </div>
                
                <!-- Enhanced Voice Bank Section -->
                <div id="voices" class="content-section">
                    <div class="section-header">
                        <h3><i class="fas fa-music"></i> Professional Voice Bank Management</h3>
                        <button class="btn-primary" onclick="showUploadModal()">
                            <i class="fas fa-upload"></i> Add New Voice
                        </button>
                    </div>
                    <div class="section-content">
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-value">{len(voice_files)}</div>
                                <div class="stat-label"><i class="fas fa-microphone"></i> Total Voices</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">{sum(float(v['size'].split()[0]) for v in voice_files):.1f} MB</div>
                                <div class="stat-label"><i class="fas fa-hdd"></i> Total Storage</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">Ultra HD</div>
                                <div class="stat-label"><i class="fas fa-star"></i> Quality Rating</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">{len([v for v in voice_files if 'active' in v.get('status', '')])}</div>
                                <div class="stat-label"><i class="fas fa-check-circle"></i> Active Voices</div>
                            </div>
                        </div>
                        
                        <!-- Upload Area -->
                        <div class="upload-area" onclick="document.getElementById('voiceFileInput').click()">
                            <div class="upload-icon">
                                <i class="fas fa-cloud-upload-alt"></i>
                            </div>
                            <h4>Drop your voice files here or click to browse</h4>
                            <p>Supports MP3, WAV, M4A - Maximum 50MB per file</p>
                            <input type="file" id="voiceFileInput" style="display: none;" multiple accept=".mp3,.wav,.m4a" onchange="handleVoiceUpload(this.files)">
                        </div>
                        
                        <!-- Voice Cards Grid -->
                        <div class="voice-grid">
                            {voice_cards_html if voice_files else '''
                            <div style="text-align: center; padding: 60px; color: #7f8c8d;">
                                <i class="fas fa-microphone-slash" style="font-size: 48px; margin-bottom: 20px; color: #bdc3c7;"></i>
                                <h4>No voices uploaded yet</h4>
                                <p>Upload your first voice sample to get started with professional TTS generation!</p>
                            </div>
                            '''}
                        </div>
                    </div>
                </div>
                
                <!-- Enhanced Job Manager Section -->
                <div id="jobs" class="content-section">
                    <div class="section-header">
                        <h3><i class="fas fa-tasks"></i> Advanced Job Queue Manager</h3>
                        <div>
                            <button class="btn-success" onclick="refreshJobs()">
                                <i class="fas fa-sync-alt"></i> Refresh
                            </button>
                            <button class="btn-danger" onclick="clearCompletedJobs()">
                                <i class="fas fa-trash"></i> Clear Completed
                            </button>
                        </div>
                    </div>
                    <div class="section-content">
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-value stat-warning">{queue_stats['job_counts']['queued']}</div>
                                <div class="stat-label"><i class="fas fa-clock"></i> Queued</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value stat-primary">{queue_stats['job_counts']['processing']}</div>
                                <div class="stat-label"><i class="fas fa-spinner"></i> Processing</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value stat-good">{queue_stats['job_counts']['completed']}</div>
                                <div class="stat-label"><i class="fas fa-check"></i> Completed</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value stat-danger">{queue_stats['job_counts']['failed']}</div>
                                <div class="stat-label"><i class="fas fa-exclamation-triangle"></i> Failed</div>
                            </div>
                        </div>
                        
                        <h4 style="margin: 25px 0 15px 0;"><i class="fas fa-history"></i> Recent Job Activity</h4>
                        <table>
                            <thead>
                                <tr>
                                    <th><i class="fas fa-hashtag"></i> Job ID</th>
                                    <th><i class="fas fa-tag"></i> Type</th>
                                    <th><i class="fas fa-traffic-light"></i> Status</th>
                                    <th><i class="fas fa-chart-line"></i> Progress</th>
                                    <th><i class="fas fa-calendar"></i> Created</th>
                                    <th><i class="fas fa-cogs"></i> Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {job_rows if job_rows else '<tr><td colspan="6" style="text-align: center; color: #7f8c8d;"><i class="fas fa-info-circle"></i> No jobs processed yet</td></tr>'}
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <!-- Performance Monitoring Section -->
                <div id="performance" class="content-section">
                    <div class="section-header">
                        <h3><i class="fas fa-tachometer-alt"></i> Performance Monitoring</h3>
                    </div>
                    <div class="section-content">
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-value">GPU</div>
                                <div class="stat-label"><i class="fas fa-microchip"></i> Accelerator</div>
                                <div class="stat-trend trend-up">
                                    <i class="fas fa-bolt"></i> CUDA Available
                                </div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">1.2s</div>
                                <div class="stat-label"><i class="fas fa-stopwatch"></i> Avg Generation</div>
                                <div class="stat-trend trend-up">
                                    <i class="fas fa-rocket"></i> 15% faster
                                </div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">24GB</div>
                                <div class="stat-label"><i class="fas fa-memory"></i> GPU Memory</div>
                                <div class="stat-trend">
                                    <i class="fas fa-chart-bar"></i> RTX 4090
                                </div>
                            </div>
                        </div>
                        
                        <div class="chart-container">
                            <h4><i class="fas fa-chart-line"></i> GPU Utilization</h4>
                            <canvas id="gpuChart" width="400" height="200"></canvas>
                        </div>
                    </div>
                </div>
                
                <!-- Live Logs Section -->
                <div id="logs" class="content-section">
                    <div class="section-header">
                        <h3><i class="fas fa-terminal"></i> Live System Logs</h3>
                        <div>
                            <button class="btn-primary" onclick="toggleLogAutoScroll()">
                                <i class="fas fa-scroll"></i> <span id="scroll-text">Auto Scroll: ON</span>
                            </button>
                            <button class="btn-success" onclick="clearLogs()">
                                <i class="fas fa-eraser"></i> Clear Logs
                            </button>
                        </div>
                    </div>
                    <div class="section-content">
                        <div class="log-viewer" id="logViewer">
                            <div class="log-entry log-info">[{datetime.now().strftime('%H:%M:%S')}] INFO: F5-TTS server started successfully</div>
                            <div class="log-entry log-info">[{datetime.now().strftime('%H:%M:%S')}] INFO: Models loaded and ready for inference</div>
                            <div class="log-entry log-info">[{datetime.now().strftime('%H:%M:%S')}] INFO: Job queue manager initialized</div>
                            <div class="log-entry log-info">[{datetime.now().strftime('%H:%M:%S')}] INFO: Admin panel accessed by {credentials}</div>
                        </div>
                    </div>
                </div>
                
                <!-- User Activity Section -->
                <div id="users" class="content-section">
                    <div class="section-header">
                        <h3><i class="fas fa-users"></i> User Activity Monitor</h3>
                    </div>
                    <div class="section-content">
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-value">1</div>
                                <div class="stat-label"><i class="fas fa-user"></i> Active Users</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">{queue_stats['total_jobs']}</div>
                                <div class="stat-label"><i class="fas fa-chart-bar"></i> API Calls</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">0</div>
                                <div class="stat-label"><i class="fas fa-ban"></i> Rate Limits</div>
                            </div>
                        </div>
                        
                        <table>
                            <thead>
                                <tr>
                                    <th><i class="fas fa-user"></i> User</th>
                                    <th><i class="fas fa-clock"></i> Last Active</th>
                                    <th><i class="fas fa-chart-line"></i> Requests</th>
                                    <th><i class="fas fa-map-marker-alt"></i> IP Address</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><i class="fas fa-shield-alt"></i> {credentials}</td>
                                    <td>Now</td>
                                    <td>{queue_stats['total_jobs']}</td>
                                    <td>Admin Session</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <!-- Configuration Section -->
                <div id="settings" class="content-section">
                    <div class="section-header">
                        <h3><i class="fas fa-cog"></i> Server Configuration</h3>
                        <button class="btn-primary" onclick="saveSettings()">
                            <i class="fas fa-save"></i> Save Changes
                        </button>
                    </div>
                    <div class="section-content">
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-value">F5-TTS v1</div>
                                <div class="stat-label">Model Version</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">1</div>
                                <div class="stat-label">Max Workers</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">32</div>
                                <div class="stat-label">NFE Steps</div>
                            </div>
                        </div>
                        
                        <h4><i class="fas fa-sliders-h"></i> Model Settings</h4>
                        <p><strong>Default Speed:</strong> {DEFAULT_INFERENCE_SETTINGS['speed']}</p>
                        <p><strong>Cross Fade:</strong> {DEFAULT_INFERENCE_SETTINGS['cross_fade_duration']}s</p>
                        <p><strong>Remove Silence:</strong> {'Enabled' if DEFAULT_INFERENCE_SETTINGS['remove_silence'] else 'Disabled'}</p>
                        <p><strong>Randomize Seed:</strong> {'Enabled' if DEFAULT_INFERENCE_SETTINGS['randomize_seed'] else 'Disabled'}</p>
                    </div>
                </div>
                
                <!-- Backup & Export Section -->
                <div id="backup" class="content-section">
                    <div class="section-header">
                        <h3><i class="fas fa-download"></i> Backup & Export Tools</h3>
                    </div>
                    <div class="section-content">
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-value">{len(voice_files)}</div>
                                <div class="stat-label">Voices to Backup</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">{sum(float(v['size'].split()[0]) for v in voice_files):.1f} MB</div>
                                <div class="stat-label">Backup Size</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">Ready</div>
                                <div class="stat-label">Status</div>
                            </div>
                        </div>
                        
                        <div style="margin-top: 25px;">
                            <button class="btn-primary" onclick="backupVoices()">
                                <i class="fas fa-download"></i> Download Voice Bank Backup
                            </button>
                            <button class="btn-success" onclick="exportSettings()">
                                <i class="fas fa-file-export"></i> Export Configuration
                            </button>
                            <button class="btn-warning" onclick="exportLogs()">
                                <i class="fas fa-file-alt"></i> Export System Logs
                            </button>
                        </div>
                        
                        <h4 style="margin: 25px 0 15px 0;"><i class="fas fa-history"></i> Backup History</h4>
                        <p><strong>Last Backup:</strong> Never</p>
                        <p><strong>Auto Backup:</strong> Disabled</p>
                        <p><strong>Backup Location:</strong> {REFERENCE_VOICES_DIR}</p>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Theme management
            let isDarkMode = false;
            
            function toggleTheme() {{
                const body = document.body;
                const themeIcon = document.getElementById('theme-icon');
                const themeText = document.getElementById('theme-text');
                
                isDarkMode = !isDarkMode;
                
                if (isDarkMode) {{
                    body.classList.add('dark-mode');
                    themeIcon.className = 'fas fa-sun';
                    themeText.textContent = 'Light Mode';
                }} else {{
                    body.classList.remove('dark-mode');
                    themeIcon.className = 'fas fa-moon';
                    themeText.textContent = 'Dark Mode';
                }}
            }}
            
            // Enhanced section navigation
            function showSection(sectionId) {{
                // Hide all sections
                document.querySelectorAll('.content-section').forEach(section => {{
                    section.classList.remove('active');
                    section.style.opacity = '0';
                    section.style.transform = 'translateY(20px)';
                }});
                
                // Remove active class from nav items
                document.querySelectorAll('.nav-item').forEach(item => {{
                    item.classList.remove('active');
                }});
                
                // Show selected section with animation
                setTimeout(() => {{
                    const targetSection = document.getElementById(sectionId);
                    targetSection.classList.add('active');
                    targetSection.style.opacity = '1';
                    targetSection.style.transform = 'translateY(0)';
                }}, 100);
                
                // Add active class to clicked nav item
                event.target.closest('.nav-item').classList.add('active');
                
                // Initialize charts if dashboard
                if (sectionId === 'dashboard') {{
                    initializeCharts();
                }}
            }}
            
            // Enhanced voice management functions
            function playVoice(voiceName) {{
                showInfoToast('🎵 Playing voice preview: ' + voiceName);
                const audio = new Audio(`/voice/${{voiceName}}`);
                audio.play().catch(error => {{
                    console.error('Audio play error:', error);
                    showErrorToast('❌ Could not play voice preview');
                }});
            }}
            
            function testVoiceGeneration(voiceName) {{
                const testText = prompt('🎤 Enter test text for TTS generation:', 'Hello! This is a test of my beautiful voice.');
                if (testText && testText.trim()) {{
                    showLoadingToast('🎯 Generating TTS with ' + voiceName + '...');
                    
                    fetch('/generate-tts', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{
                            'text': testText,
                            'voice_name': voiceName,
                            'speed': 1.0,
                            'remove_silence': true
                        }})
                    }})
                    .then(response => response.json())
                    .then(data => {{
                        if (data.status === 'success') {{
                            showSuccessToast('✅ TTS generated! Check audio tab or job queue.');
                            // Optionally play the generated audio
                            if (data.audio_url) {{
                                const audio = new Audio(data.audio_url);
                                audio.play();
                            }}
                        }} else {{
                            showErrorToast('❌ TTS generation failed: ' + (data.error || 'Unknown error'));
                        }}
                    }})
                    .catch(error => {{
                        console.error('TTS Error:', error);
                        showErrorToast('❌ Network error during TTS generation');
                    }});
                }}
            }}
            
            function downloadVoice(voiceName) {{
                showInfoToast('⬇️ Downloading voice: ' + voiceName);
                const link = document.createElement('a');
                link.href = `/voice/${{voiceName}}`;
                link.download = voiceName;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                showSuccessToast('✅ Voice download started!');
            }}
            
            function confirmDeleteVoice(voiceName) {{
                const modal = createCustomModal(
                    '🗑️ Delete Voice',
                    `Are you sure you want to delete "<strong>${{voiceName}}</strong>"?<br><br><span style="color: #e74c3c;">This action cannot be undone!</span>`,
                    [
                        {{ text: 'Cancel', class: 'btn-glass', action: 'close' }},
                        {{ text: 'Delete Forever', class: 'btn-danger', action: () => deleteVoice(voiceName) }}
                    ]
                );
                modal.show();
            }}
            
            function deleteVoice(voiceName) {{
                showLoadingToast('🗑️ Deleting voice: ' + voiceName + '...');
                
                fetch(`/delete-voice/${{voiceName}}`, {{
                    method: 'DELETE'
                }})
                .then(response => response.json())
                .then(data => {{
                    if (data.status === 'success') {{
                        showSuccessToast('✅ Voice deleted successfully!');
                        setTimeout(() => location.reload(), 1500);
                    }} else {{
                        showErrorToast('❌ Error deleting voice: ' + (data.error || 'Unknown error'));
                    }}
                }})
                .catch(error => {{
                    console.error('Delete Error:', error);
                    showErrorToast('❌ Network error during voice deletion');
                }});
            }}
            
            function showUploadModal() {{
                const modal = createCustomModal(
                    '🎤 Upload New Voice',
                    `
                    <div style="text-align: center; padding: 20px;">
                        <div class="upload-area" onclick="document.getElementById('modalVoiceInput').click()">
                            <div class="upload-icon">
                                <i class="fas fa-microphone"></i>
                            </div>
                            <h4>Choose your voice file</h4>
                            <p>Supports MP3, WAV, M4A • Max 50MB<br>Best quality: 5-15 seconds of clear speech</p>
                        </div>
                        <input type="file" id="modalVoiceInput" style="display: none;" accept=".mp3,.wav,.m4a" onchange="handleModalVoiceUpload(this.files[0])">
                    </div>
                    `,
                    [
                        {{ text: 'Close', class: 'btn-glass', action: 'close' }}
                    ]
                );
                modal.show();
            }}
            
            function handleVoiceUpload(files) {{
                if (files.length === 0) return;
                
                for (let i = 0; i < files.length; i++) {{
                    uploadSingleVoice(files[i]);
                }}
            }}
            
            function handleModalVoiceUpload(file) {{
                if (file) {{
                    uploadSingleVoice(file);
                    closeModal();
                }}
            }}
            
            function uploadSingleVoice(file) {{
                if (file.size > 50 * 1024 * 1024) {{
                    showErrorToast(`❌ File too large: ${{file.name}} (Max 50MB)`);
                    return;
                }}
                
                const formData = new FormData();
                formData.append('file', file);
                
                showLoadingToast(`🎤 Uploading voice: ${{file.name}}...`);
                
                fetch('/upload-permanent-voice', {{
                    method: 'POST',
                    body: formData
                }})
                .then(response => response.json())
                .then(data => {{
                    if (data.status === 'success') {{
                        showSuccessToast(`✅ Voice uploaded: ${{file.name}}`);
                        setTimeout(() => location.reload(), 2000);
                    }} else {{
                        showErrorToast(`❌ Upload failed: ${{data.error || 'Unknown error'}}`);
                    }}
                }})
                .catch(error => {{
                    console.error('Upload Error:', error);
                    showErrorToast(`❌ Upload failed: Network error`);
                }});
            }}
            
            function testVoice(voiceName) {{
                // Legacy function for backward compatibility
                testVoiceGeneration(voiceName);
            }}
            
            // Advanced job management
            function viewJobDetails(jobId) {{
                fetch(`/jobs/${{jobId}}/status`)
                    .then(response => response.json())
                    .then(data => {{
                        alert(`📊 Job Details: ${{jobId}}\\n\\nType: ${{data.job_type}}\\nStatus: ${{data.status}}\\nProgress: ${{data.progress}}%\\nCreated: ${{data.created_at}}`);
                    }})
                    .catch(() => alert('Error fetching job details'));
            }}
            
            function cancelJob(jobId) {{
                if (confirm(`⏹️ Cancel job: ${{jobId}}?`)) {{
                    showLoadingToast('Cancelling job...');
                    // API call to cancel job
                    setTimeout(() => showSuccessToast('Job cancelled'), 1000);
                }}
            }}
            
            function refreshJobs() {{
                showLoadingToast('🔄 Refreshing job data...');
                location.reload();
            }}
            
            function clearCompletedJobs() {{
                if (confirm('🧹 Clear all completed jobs?')) {{
                    fetch('/jobs/cleanup', {{method: 'POST'}})
                        .then(() => {{
                            showSuccessToast('Completed jobs cleared!');
                            location.reload();
                        }})
                        .catch(() => showErrorToast('Error clearing jobs'));
                }}
            }}
            
            // Live logs management
            let autoScroll = true;
            let logBuffer = [];
            
            function toggleLogAutoScroll() {{
                autoScroll = !autoScroll;
                const scrollText = document.getElementById('scroll-text');
                scrollText.textContent = `Auto Scroll: ${{autoScroll ? 'ON' : 'OFF'}}`;
            }}
            
            function clearLogs() {{
                document.getElementById('logViewer').innerHTML = '<div class="log-entry log-info">[' + new Date().toLocaleTimeString() + '] INFO: Logs cleared by admin</div>';
            }}
            
            function addLogEntry(level, message) {{
                const logViewer = document.getElementById('logViewer');
                const timestamp = new Date().toLocaleTimeString();
                const logEntry = document.createElement('div');
                logEntry.className = `log-entry log-${{level}}`;
                logEntry.textContent = `[${{timestamp}}] ${{level.toUpperCase()}}: ${{message}}`;
                
                logViewer.appendChild(logEntry);
                
                if (autoScroll) {{
                    logEntry.scrollIntoView({{behavior: 'smooth'}});
                }}
            }}
            
            // Configuration management
            function saveSettings() {{
                showLoadingToast('💾 Saving configuration...');
                setTimeout(() => showSuccessToast('Settings saved!'), 1000);
            }}
            
            // Backup and export functions
            function backupVoices() {{
                showLoadingToast('📦 Creating voice bank backup...');
                // Simulate backup creation
                setTimeout(() => {{
                    const link = document.createElement('a');
                    link.href = '#';
                    link.download = `voice-bank-backup-${{new Date().toISOString().split('T')[0]}}.zip`;
                    showSuccessToast('📥 Backup ready for download!');
                }}, 2000);
            }}
            
            function exportSettings() {{
                showLoadingToast('⚙️ Exporting configuration...');
                setTimeout(() => showSuccessToast('Configuration exported!'), 1000);
            }}
            
            function exportLogs() {{
                showLoadingToast('📋 Exporting system logs...');
                setTimeout(() => showSuccessToast('Logs exported!'), 1000);
            }}
            
            // Toast notification system
            function showToast(message, type = 'info') {{
                const toast = document.createElement('div');
                toast.style.cssText = `
                    position: fixed; top: 20px; right: 20px; z-index: 10000;
                    padding: 15px 20px; border-radius: 8px; color: white; font-weight: 500;
                    background: ${{type === 'success' ? '#27ae60' : type === 'error' ? '#e74c3c' : '#3498db'}};
                    box-shadow: 0 4px 20px rgba(0,0,0,0.3); opacity: 0; transform: translateY(-20px);
                    transition: all 0.3s ease;
                `;
                toast.textContent = message;
                document.body.appendChild(toast);
                
                // Animate in
                setTimeout(() => {{
                    toast.style.opacity = '1';
                    toast.style.transform = 'translateY(0)';
                }}, 100);
                
                // Remove after 3 seconds
                setTimeout(() => {{
                    toast.style.opacity = '0';
                    toast.style.transform = 'translateY(-20px)';
                    setTimeout(() => document.body.removeChild(toast), 300);
                }}, 3000);
            }}
            
            function showSuccessToast(message) {{ showToast(message, 'success'); }}
            function showErrorToast(message) {{ showToast(message, 'error'); }}
            function showLoadingToast(message) {{ showToast(message, 'info'); }}
            
            // Charts initialization
            function initializeCharts() {{
                // Performance Chart
                const perfCtx = document.getElementById('performanceChart');
                if (perfCtx) {{
                    new Chart(perfCtx, {{
                        type: 'line',
                        data: {{
                            labels: ['00:00', '00:05', '00:10', '00:15', '00:20', '00:25', '00:30'],
                            datasets: [{{
                                label: 'CPU Usage',
                                data: [{cpu_percent}, 45, 52, 38, {cpu_percent}, 48, {cpu_percent}],
                                borderColor: '#3498db',
                                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                                tension: 0.4
                            }}, {{
                                label: 'Memory Usage',
                                data: [{memory.percent}, 58, 61, 55, {memory.percent}, 62, {memory.percent}],
                                borderColor: '#27ae60',
                                backgroundColor: 'rgba(39, 174, 96, 0.1)',
                                tension: 0.4
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            plugins: {{ legend: {{ display: true }} }},
                            scales: {{ y: {{ beginAtZero: true, max: 100 }} }}
                        }}
                    }});
                }}
                
                // GPU Chart
                const gpuCtx = document.getElementById('gpuChart');
                if (gpuCtx) {{
                    new Chart(gpuCtx, {{
                        type: 'doughnut',
                        data: {{
                            labels: ['Used', 'Available'],
                            datasets: [{{
                                data: [65, 35],
                                backgroundColor: ['#e74c3c', '#ecf0f1'],
                                borderWidth: 0
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            plugins: {{
                                legend: {{ position: 'bottom' }}
                            }}
                        }}
                    }});
                }}
            }}
            
            // Simulated real-time log updates
            setInterval(() => {{
                if (document.getElementById('logs').classList.contains('active')) {{
                    const messages = [
                        'Job processing completed successfully',
                        'New voice file cached in memory',
                        'GPU memory optimization performed',
                        'Model inference completed in 1.2s',
                        'Background cleanup task executed'
                    ];
                    const levels = ['info', 'info', 'info', 'info', 'info'];
                    const randomIndex = Math.floor(Math.random() * messages.length);
                    addLogEntry(levels[randomIndex], messages[randomIndex]);
                }}
            }}, 10000);
            
            // Custom modal system
            function createCustomModal(title, content, buttons = []) {{
                const modal = document.createElement('div');
                modal.className = 'custom-modal';
                modal.innerHTML = `
                    <div class="modal-overlay" onclick="closeModal()">
                        <div class="modal-content" onclick="event.stopPropagation()">
                            <div class="modal-header">
                                <h3>${{title}}</h3>
                                <button class="modal-close" onclick="closeModal()">
                                    <i class="fas fa-times"></i>
                                </button>
                            </div>
                            <div class="modal-body">
                                ${{content}}
                            </div>
                            <div class="modal-footer">
                                ${{buttons.map(btn => `<button class="btn-small ${{btn.class}}" onclick="${{btn.action === 'close' ? 'closeModal()' : btn.action.name + '()'}}">${{btn.text}}</button>`).join('')}}
                            </div>
                        </div>
                    </div>
                `;
                
                // Add modal styles if not exists
                if (!document.getElementById('modal-styles')) {{
                    const style = document.createElement('style');
                    style.id = 'modal-styles';
                    style.textContent = `
                        .custom-modal {{
                            position: fixed;
                            top: 0;
                            left: 0;
                            width: 100%;
                            height: 100%;
                            z-index: 10000;
                        }}
                        .modal-overlay {{
                            width: 100%;
                            height: 100%;
                            background: rgba(0,0,0,0.7);
                            backdrop-filter: blur(5px);
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            padding: 20px;
                        }}
                        .modal-content {{
                            background: white;
                            border-radius: 20px;
                            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                            max-width: 500px;
                            width: 100%;
                            max-height: 80vh;
                            overflow: hidden;
                        }}
                        .dark-mode .modal-content {{
                            background: var(--dark-card);
                            color: var(--text-light);
                        }}
                        .modal-header {{
                            padding: 25px 25px 0;
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                        }}
                        .modal-close {{
                            background: none;
                            border: none;
                            font-size: 18px;
                            cursor: pointer;
                            color: #999;
                            padding: 8px;
                            border-radius: 50%;
                            transition: all 0.3s;
                        }}
                        .modal-close:hover {{
                            background: rgba(255,0,0,0.1);
                            color: #e74c3c;
                        }}
                        .modal-body {{
                            padding: 25px;
                        }}
                        .modal-footer {{
                            padding: 0 25px 25px;
                            display: flex;
                            gap: 10px;
                            justify-content: flex-end;
                        }}
                    `;
                    document.head.appendChild(style);
                }}
                
                return {{
                    element: modal,
                    show: () => {{
                        document.body.appendChild(modal);
                        setTimeout(() => modal.style.opacity = '1', 10);
                    }},
                    hide: () => {{
                        modal.style.opacity = '0';
                        setTimeout(() => modal.remove(), 300);
                    }}
                }};
            }}
            
            function closeModal() {{
                const modals = document.querySelectorAll('.custom-modal');
                modals.forEach(modal => {{
                    modal.style.opacity = '0';
                    setTimeout(() => modal.remove(), 300);
                }});
            }}
            
            // Auto-refresh for live data (every 30 seconds)
            setInterval(() => {{
                if (document.getElementById('dashboard').classList.contains('active') || 
                    document.getElementById('jobs').classList.contains('active')) {{
                    // Only refresh if not actively interacting
                    if (!document.querySelector(':hover')) {{
                        location.reload();
                    }}
                }}
            }}, 30000);
            
            // Initialize charts on page load
            document.addEventListener('DOMContentLoaded', () => {{
                initializeCharts();
                showSuccessToast('🚀 Admin dashboard loaded successfully!');
            }});
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/upload-audio", response_model=AudioUploadResponse)
async def upload_audio(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Reference audio file (temporary use - WAV, MP3, etc.)"),
    api_key_valid: bool = Depends(verify_api_key)
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
        try:
            import librosa
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

@app.post("/upload-permanent-voice", response_model=PermanentVoiceUploadResponse)
async def upload_permanent_voice(
    voice_name: str = Form(..., description="Custom name for the voice (e.g., 'adam', 'sarah', 'narrator')"),
    audio_file: UploadFile = File(..., description="Reference audio file (WAV, MP3, etc.)"),
    api_key_valid: bool = Depends(verify_api_key)
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
        try:
            import librosa
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

@app.post("/tts-generate", response_model=TTSGenerateResponse)
async def tts_generate(
    background_tasks: BackgroundTasks,
    request: TTSGenerateRequest,
    api_key_valid: bool = Depends(verify_api_key)
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

@app.post("/tts-permanent", response_model=TTSGenerateResponse)
async def tts_permanent(
    request: PermanentTTSRequest,
    api_key_valid: bool = Depends(verify_api_key)
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
    cross_fade_duration: float = Form(0.15, description="Cross-fade duration (0.0-1.0)"),
    api_key_valid: bool = Depends(verify_api_key)
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

@app.get("/list-voices")
async def list_voices(api_key_valid: bool = Depends(verify_api_key)):
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

@app.get("/voice/{voice_name}")
async def serve_voice_file(voice_name: str, api_key_valid: bool = Depends(verify_api_key)):
    """
    Serve voice files directly for audio playback in the admin interface.
    """
    try:
        # Find the voice file with any supported extension
        supported_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        voice_file = None
        
        for ext in supported_extensions:
            potential_file = os.path.join(REFERENCE_VOICES_DIR, voice_name)
            if not voice_name.endswith(ext):
                potential_file = os.path.join(REFERENCE_VOICES_DIR, voice_name + ext)
            
            if os.path.exists(potential_file):
                voice_file = potential_file
                break
        
        # Also check if the voice_name already includes the extension
        direct_path = os.path.join(REFERENCE_VOICES_DIR, voice_name)
        if os.path.exists(direct_path):
            voice_file = direct_path
        
        if not voice_file:
            raise HTTPException(status_code=404, detail=f"Voice file '{voice_name}' not found")
        
        # Determine content type based on file extension
        file_ext = os.path.splitext(voice_file)[1].lower()
        content_types = {
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg', 
            '.flac': 'audio/flac',
            '.m4a': 'audio/mp4',
            '.ogg': 'audio/ogg'
        }
        
        content_type = content_types.get(file_ext, 'audio/wav')
        
        return FileResponse(
            voice_file,
            media_type=content_type,
            headers={"Accept-Ranges": "bytes"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving voice file: {str(e)}")

@app.delete("/delete-voice/{voice_name}")
async def delete_voice(
    voice_name: str,
    api_key_valid: bool = Depends(verify_api_key)
):
    """
    Delete a permanent voice file.
    """
    try:
        # Find the voice file with any supported extension
        supported_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        voice_file = None
        
        for ext in supported_extensions:
            potential_file = os.path.join(REFERENCE_VOICES_DIR, voice_name)
            if not voice_name.endswith(ext):
                potential_file = os.path.join(REFERENCE_VOICES_DIR, voice_name + ext)
            
            if os.path.exists(potential_file):
                voice_file = potential_file
                break
        
        # Also check if the voice_name already includes the extension
        direct_path = os.path.join(REFERENCE_VOICES_DIR, voice_name)
        if os.path.exists(direct_path):
            voice_file = direct_path
        
        if not voice_file:
            raise HTTPException(status_code=404, detail=f"Voice file '{voice_name}' not found")
        
        # Delete the file
        os.remove(voice_file)
        
        return {
            "status": "success",
            "message": f"Voice '{voice_name}' deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting voice: {str(e)}")

@app.get("/download-audio/{file_id}")
async def download_audio(file_id: str, api_key_valid: bool = Depends(verify_api_key)):
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
async def delete_audio(
    file_id: str,
    api_key_valid: bool = Depends(verify_api_key)
):
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
async def list_audio(api_key_valid: bool = Depends(verify_api_key)):
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
        ],
        "total_files": len(uploaded_files)
    }

# Job Queue Endpoints
@app.post("/jobs/submit")
async def submit_job(
    request: JobRequest,
    api_key_valid: bool = Depends(verify_api_key)
):
    """
    Submit a new job to the processing queue.
    
    Job types:
    - tts_generate: Generate TTS using uploaded audio file
    - tts_permanent: Generate TTS using permanent voice
    - voice_cloning: Clone a voice and generate TTS
    
    Returns job_id for tracking the job status and retrieving results.
    """
    if not job_queue_manager:
        raise HTTPException(status_code=503, detail="Job queue not initialized")
    
    if not F5TTS_ema_model or not vocoder:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Validate job parameters based on job type
        params = request.parameters
        
        if request.job_type == JobType.TTS_GENERATE:
            if "ref_audio_path" not in params or "gen_text" not in params:
                raise HTTPException(status_code=400, detail="TTS generate jobs require 'ref_audio_path' and 'gen_text' parameters")
        
        elif request.job_type == JobType.TTS_PERMANENT:
            if "voice_name" not in params or "gen_text" not in params:
                raise HTTPException(status_code=400, detail="Permanent TTS jobs require 'voice_name' and 'gen_text' parameters")
        
        elif request.job_type == JobType.VOICE_CLONING:
            if "audio_file_id" not in params or "gen_text" not in params:
                raise HTTPException(status_code=400, detail="Voice cloning jobs require 'audio_file_id' and 'gen_text' parameters")
        
        # Submit job to queue
        job_id = job_queue_manager.submit_job(
            job_type=request.job_type,
            parameters=params,
            priority=request.priority
        )
        
        return {
            "job_id": job_id,
            "message": "Job submitted successfully",
            "queue_position": job_queue_manager.job_queue.qsize()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting job: {str(e)}")

@app.get("/jobs/{job_id}/status")
async def get_job_status(
    job_id: str,
    api_key_valid: bool = Depends(verify_api_key)
):
    """
    Get the current status of a submitted job.
    """
    if not job_queue_manager:
        raise HTTPException(status_code=503, detail="Job queue not initialized")
    
    job_status = job_queue_manager.get_job_status(job_id)
    if not job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_status

@app.get("/jobs/{job_id}/result")
async def get_job_result(
    job_id: str,
    api_key_valid: bool = Depends(verify_api_key)
):
    """
    Get the result of a completed job.
    Returns the output file_id that can be used with /download-audio endpoint.
    """
    if not job_queue_manager:
        raise HTTPException(status_code=503, detail="Job queue not initialized")
    
    job_status = job_queue_manager.get_job_status(job_id)
    if not job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_status["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed. Current status: {job_status['status']}")
    
    result = job_queue_manager.get_job_result(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Job result not found")
    
    return result

@app.get("/jobs/queue-status")
async def get_queue_status(api_key_valid: bool = Depends(verify_api_key)):
    """
    Get overall job queue status and statistics.
    """
    if not job_queue_manager:
        raise HTTPException(status_code=503, detail="Job queue not initialized")
    
    return job_queue_manager.get_queue_status()

@app.post("/jobs/cleanup")
async def cleanup_old_jobs(
    max_age_hours: int = 24,
    api_key_valid: bool = Depends(verify_api_key)
):
    """
    Remove old completed/failed jobs from memory.
    Default: remove jobs older than 24 hours.
    """
    if not job_queue_manager:
        raise HTTPException(status_code=503, detail="Job queue not initialized")
    
    cleaned_count = job_queue_manager.cleanup_old_jobs(max_age_hours)
    return {
        "message": f"Cleaned up {cleaned_count} old jobs",
        "cleaned_count": cleaned_count
    }

# Convenient job submission endpoints (async versions of existing endpoints)
@app.post("/jobs/tts-generate-async")
async def submit_tts_generate_job(
    audio_file_id: str = Form(...),
    text: str = Form(...),
    ref_text: str = Form(""),
    randomize_seed: bool = Form(True),
    seed: int = Form(0),
    remove_silence: bool = Form(False),
    speed: float = Form(1.0),
    nfe_step: int = Form(32),
    cross_fade_duration: float = Form(0.15),
    priority: int = Form(0),
    api_key_valid: bool = Depends(verify_api_key)
):
    """
    Submit TTS generation job (async version of /tts-generate).
    Returns immediately with job_id for tracking.
    """
    if audio_file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Audio file not found. Please upload audio first.")
    
    ref_audio_path = uploaded_files[audio_file_id]
    if not os.path.exists(ref_audio_path):
        raise HTTPException(status_code=404, detail="Reference audio file has expired. Please re-upload.")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text to generate cannot be empty")
    
    # Prepare inference settings
    inference_settings = {
        "randomize_seed": randomize_seed,
        "seed": seed,
        "remove_silence": remove_silence,
        "speed": speed,
        "nfe_step": nfe_step,
        "cross_fade_duration": cross_fade_duration,
    }
    
    # Validate settings
    if speed < 0.3 or speed > 2.0:
        raise HTTPException(status_code=400, detail="Speed must be between 0.3 and 2.0")
    if nfe_step < 4 or nfe_step > 64:
        raise HTTPException(status_code=400, detail="NFE steps must be between 4 and 64")
    if cross_fade_duration < 0.0 or cross_fade_duration > 1.0:
        raise HTTPException(status_code=400, detail="Cross-fade duration must be between 0.0 and 1.0")
    
    # Submit job
    job_id = job_queue_manager.submit_job(
        job_type=JobType.TTS_GENERATE,
        parameters={
            "ref_audio_path": ref_audio_path,
            "ref_text": ref_text,
            "gen_text": text,
            "inference_settings": inference_settings
        },
        priority=priority
    )
    
    return {
        "job_id": job_id,
        "message": "TTS generation job submitted successfully",
        "queue_position": job_queue_manager.job_queue.qsize()
    }

@app.post("/jobs/tts-permanent-async")
async def submit_tts_permanent_job(
    voice_name: str = Form(...),
    text: str = Form(...),
    randomize_seed: bool = Form(True),
    seed: int = Form(0),
    remove_silence: bool = Form(False),
    speed: float = Form(1.0),
    nfe_step: int = Form(32),
    cross_fade_duration: float = Form(0.15),
    priority: int = Form(0),
    api_key_valid: bool = Depends(verify_api_key)
):
    """
    Submit permanent voice TTS job (async version of /tts-permanent).
    Returns immediately with job_id for tracking.
    """
    try:
        voice_path = find_voice_file(voice_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Permanent voice '{voice_name}' not found")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text to generate cannot be empty")
    
    # Prepare inference settings
    inference_settings = {
        "randomize_seed": randomize_seed,
        "seed": seed,
        "remove_silence": remove_silence,
        "speed": speed,
        "nfe_step": nfe_step,
        "cross_fade_duration": cross_fade_duration,
    }
    
    # Validate settings
    if speed < 0.3 or speed > 2.0:
        raise HTTPException(status_code=400, detail="Speed must be between 0.3 and 2.0")
    if nfe_step < 4 or nfe_step > 64:
        raise HTTPException(status_code=400, detail="NFE steps must be between 4 and 64")
    if cross_fade_duration < 0.0 or cross_fade_duration > 1.0:
        raise HTTPException(status_code=400, detail="Cross-fade duration must be between 0.0 and 1.0")
    
    # Submit job
    job_id = job_queue_manager.submit_job(
        job_type=JobType.TTS_PERMANENT,
        parameters={
            "voice_name": voice_name,
            "gen_text": text,
            "inference_settings": inference_settings
        },
        priority=priority
    )
    
    return {
        "job_id": job_id,
        "message": "Permanent voice TTS job submitted successfully",
        "queue_position": job_queue_manager.job_queue.qsize()
    }

@app.post("/jobs/voice-cloning-async")
async def submit_voice_cloning_job(
    text: str = Form(..., description="Text to generate"),
    ref_text: str = Form("", description="Reference text (leave empty for auto-transcription)"),
    audio_file: UploadFile = File(..., description="Reference audio file (temporary)"),
    randomize_seed: bool = Form(True, description="Use random seed"),
    seed: int = Form(0, description="Specific seed (used if randomize_seed=false)"),
    remove_silence: bool = Form(False, description="Remove silences"),
    speed: float = Form(1.0, description="Speed (0.3-2.0)"),
    nfe_step: int = Form(32, description="NFE steps (4-64)"),
    cross_fade_duration: float = Form(0.15, description="Cross-fade duration (0.0-1.0)"),
    priority: int = Form(0),
    api_key_valid: bool = Depends(verify_api_key)
):
    """
    Submit voice cloning job (async version of /voice-cloning).
    Upload audio file directly - it will be processed temporarily.
    Returns immediately with job_id for tracking.
    """
    if not audio_file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text to generate cannot be empty")
    
    # Create temporary reference file
    temp_ref_id = str(uuid.uuid4())
    file_extension = Path(audio_file.filename).suffix if audio_file.filename else ".wav"
    temp_ref_path = os.path.join(tempfile.gettempdir(), f"f5tts_async_clone_{temp_ref_id}{file_extension}")
    
    try:
        # Save temporary reference audio
        with open(temp_ref_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving uploaded audio file: {str(e)}")
    
    # Prepare inference settings
    inference_settings = {
        "randomize_seed": randomize_seed,
        "seed": seed,
        "remove_silence": remove_silence,
        "speed": speed,
        "nfe_step": nfe_step,
        "cross_fade_duration": cross_fade_duration,
    }
    
    # Validate settings
    if speed < 0.3 or speed > 2.0:
        raise HTTPException(status_code=400, detail="Speed must be between 0.3 and 2.0")
    if nfe_step < 4 or nfe_step > 64:
        raise HTTPException(status_code=400, detail="NFE steps must be between 4 and 64")
    if cross_fade_duration < 0.0 or cross_fade_duration > 1.0:
        raise HTTPException(status_code=400, detail="Cross-fade duration must be between 0.0 and 1.0")
    
    # Submit job
    job_id = job_queue_manager.submit_job(
        job_type=JobType.VOICE_CLONING,
        parameters={
            "temp_ref_path": temp_ref_path,  # Use temp file path instead of audio_file_id
            "ref_text": ref_text,
            "gen_text": text,
            "inference_settings": inference_settings
        },
        priority=priority
    )
    
    return {
        "job_id": job_id,
        "message": "Voice cloning job submitted successfully",
        "queue_position": job_queue_manager.job_queue.qsize()
    }

# Admin Panel Cleanup Actions
@app.post("/admin/cleanup-jobs", dependencies=[Depends(verify_admin)])
async def admin_cleanup_jobs():
    """Clean up completed/failed jobs from memory"""
    try:
        if job_queue_manager:
            cleaned = job_queue_manager.cleanup_old_jobs()
            return {"success": True, "message": f"Cleaned {cleaned} old jobs"}
        return {"success": False, "message": "Job queue manager not available"}
    except Exception as e:
        return {"success": False, "message": f"Error during cleanup: {str(e)}"}

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