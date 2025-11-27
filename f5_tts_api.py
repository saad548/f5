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
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Request, Depends
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import uvicorn
import secrets
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

@app.post("/tts-permanent", response_model=TTSGenerateResponse)
async def tts_permanent(request: PermanentTTSRequest):
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
        ],
        "total_files": len(uploaded_files)
    }

# Job Queue Endpoints
@app.post("/jobs/submit")
async def submit_job(request: JobRequest):
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
async def get_job_status(job_id: str):
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
async def get_job_result(job_id: str):
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
async def get_queue_status():
    """
    Get overall job queue status and statistics.
    """
    if not job_queue_manager:
        raise HTTPException(status_code=503, detail="Job queue not initialized")
    
    return job_queue_manager.get_queue_status()

@app.post("/jobs/cleanup")
async def cleanup_old_jobs(max_age_hours: int = 24):
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
    priority: int = Form(0)
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
    priority: int = Form(0)
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
    audio_file_id: str = Form(...),
    text: str = Form(...),
    randomize_seed: bool = Form(True),
    seed: int = Form(0),
    remove_silence: bool = Form(False),
    speed: float = Form(1.0),
    nfe_step: int = Form(32),
    cross_fade_duration: float = Form(0.15),
    priority: int = Form(0)
):
    """
    Submit voice cloning job (async version of /voice-cloning).
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
        job_type=JobType.VOICE_CLONING,
        parameters={
            "audio_file_id": audio_file_id,
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

# Admin Panel Endpoints
@app.get("/admin", response_class=HTMLResponse, dependencies=[Depends(verify_admin)])
async def admin_panel():
    """Admin panel homepage with system overview"""
    
    # Get system stats
    try:
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
    except:
        cpu_percent = 0
        memory = type('obj', (object,), {'percent': 0, 'used': 0, 'total': 0})
        disk = type('obj', (object,), {'percent': 0, 'used': 0, 'total': 0})
    
    # Get voice counts
    try:
        voice_files = [f for f in os.listdir(REFERENCE_VOICES_DIR) if not f.startswith('.')]
        permanent_voices = len([f for f in voice_files if f.endswith(('.wav', '.mp3', '.flac'))])
    except:
        permanent_voices = 0
    
    # Get queue stats
    queue_stats = {"total_jobs": 0, "queue_size": 0, "current_job": None}
    if job_queue_manager:
        queue_stats = job_queue_manager.get_queue_status()
    
    # Get uploaded files count
    uploaded_count = len(uploaded_files)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>F5-TTS Admin Panel</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
            .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }}
            .stat-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .stat-title {{ font-size: 14px; color: #666; margin-bottom: 5px; }}
            .stat-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .progress-bar {{ width: 100%; height: 8px; background: #eee; border-radius: 4px; margin-top: 5px; }}
            .progress-fill {{ height: 100%; border-radius: 4px; }}
            .nav {{ margin-bottom: 20px; }}
            .nav a {{ display: inline-block; padding: 10px 20px; background: #3498db; color: white; text-decoration: none; border-radius: 4px; margin-right: 10px; }}
            .nav a:hover {{ background: #2980b9; }}
            .status-good {{ color: #27ae60; }}
            .status-warning {{ color: #f39c12; }}
            .status-error {{ color: #e74c3c; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎤 F5-TTS Admin Panel</h1>
                <p>System monitoring and voice management dashboard</p>
            </div>
            
            <div class="nav">
                <a href="/admin">Dashboard</a>
                <a href="/admin/voices">Voice Management</a>
                <a href="/admin/jobs">Job Queue</a>
                <a href="/admin/files">File Manager</a>
                <a href="/admin/docs">API Docs</a>
                <a href="/admin/logs">System Logs</a>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-title">System Status</div>
                    <div class="stat-value status-good">🟢 Online</div>
                    <div>Platform: {platform.system()} {platform.release()}</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-title">CPU Usage</div>
                    <div class="stat-value">{cpu_percent:.1f}%</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {cpu_percent}%; background: {'#e74c3c' if cpu_percent > 80 else '#f39c12' if cpu_percent > 50 else '#27ae60'};"></div>
                    </div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-title">Memory Usage</div>
                    <div class="stat-value">{memory.percent:.1f}%</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {memory.percent}%; background: {'#e74c3c' if memory.percent > 80 else '#f39c12' if memory.percent > 50 else '#27ae60'};"></div>
                    </div>
                    <div>{memory.used // (1024**3):.1f} GB / {memory.total // (1024**3):.1f} GB</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-title">Disk Usage</div>
                    <div class="stat-value">{disk.percent:.1f}%</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {disk.percent}%; background: {'#e74c3c' if disk.percent > 80 else '#f39c12' if disk.percent > 50 else '#27ae60'};"></div>
                    </div>
                    <div>{disk.used // (1024**3):.1f} GB / {disk.total // (1024**3):.1f} GB</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-title">Permanent Voices</div>
                    <div class="stat-value">{permanent_voices}</div>
                    <div>Reference voices available</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-title">Uploaded Files</div>
                    <div class="stat-value">{uploaded_count}</div>
                    <div>Temporary audio files</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-title">Job Queue</div>
                    <div class="stat-value">{queue_stats['total_jobs']}</div>
                    <div>Total: {queue_stats['total_jobs']} | Queue: {queue_stats['queue_size']}</div>
                    <div>Current: {queue_stats['current_job'][:8] if queue_stats['current_job'] else 'None'}</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-title">Model Status</div>
                    <div class="stat-value {'status-good' if F5TTS_ema_model and vocoder else 'status-error'}">{'🟢 Loaded' if F5TTS_ema_model and vocoder else '🔴 Not Loaded'}</div>
                    <div>F5-TTS: {'✓' if F5TTS_ema_model else '✗'} | Vocoder: {'✓' if vocoder else '✗'}</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

@app.get("/admin/voices", response_class=HTMLResponse, dependencies=[Depends(verify_admin)])
async def admin_voices():
    """Voice management page"""
    
    try:
        voice_files = os.listdir(REFERENCE_VOICES_DIR)
        voices_data = []
        
        for file in voice_files:
            if file.startswith('.'):
                continue
                
            file_path = os.path.join(REFERENCE_VOICES_DIR, file)
            if os.path.isfile(file_path):
                try:
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    mod_time = os.path.getmtime(file_path)
                    
                    voices_data.append({
                        'name': file,
                        'size': f"{file_size:.2f} MB",
                        'modified': time.strftime('%Y-%m-%d %H:%M', time.localtime(mod_time)),
                        'type': file.split('.')[-1].upper() if '.' in file else 'Unknown'
                    })
                except:
                    continue
                    
    except Exception as e:
        voices_data = []
    
    voice_rows = ""
    for voice in voices_data:
        voice_rows += f"""
        <tr>
            <td>{voice['name']}</td>
            <td>{voice['type']}</td>
            <td>{voice['size']}</td>
            <td>{voice['modified']}</td>
            <td>
                <button onclick="testVoice('{voice['name']}')">Test</button>
                <button onclick="deleteVoice('{voice['name']}')">Delete</button>
            </td>
        </tr>
        """
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voice Management - F5-TTS Admin</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
            .nav {{ margin-bottom: 20px; }}
            .nav a {{ display: inline-block; padding: 10px 20px; background: #3498db; color: white; text-decoration: none; border-radius: 4px; margin-right: 10px; }}
            .nav a:hover {{ background: #2980b9; }}
            table {{ width: 100%; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
            th {{ background: #34495e; color: white; }}
            button {{ padding: 5px 10px; margin: 2px; border: none; border-radius: 3px; cursor: pointer; }}
            .test-btn {{ background: #27ae60; color: white; }}
            .delete-btn {{ background: #e74c3c; color: white; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎤 Voice Management</h1>
                <p>Manage permanent reference voices</p>
            </div>
            
            <div class="nav">
                <a href="/admin">Dashboard</a>
                <a href="/admin/voices">Voice Management</a>
                <a href="/admin/jobs">Job Queue</a>
                <a href="/admin/files">File Manager</a>
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>Voice Name</th>
                        <th>Type</th>
                        <th>Size</th>
                        <th>Modified</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {voice_rows}
                </tbody>
            </table>
        </div>
        
        <script>
            function testVoice(voiceName) {{
                alert('Testing voice: ' + voiceName + '\n\nFeature coming soon!');
            }}
            
            function deleteVoice(voiceName) {{
                if (confirm('Are you sure you want to delete voice: ' + voiceName + '?')) {{
                    alert('Delete functionality coming soon!');
                }}
            }}
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/admin/jobs", dependencies=[Depends(verify_admin)])
async def admin_jobs():
    """Job queue management"""
    if not job_queue_manager:
        raise HTTPException(503, "Job queue not initialized")
    
    queue_status = job_queue_manager.get_queue_status()
    
    # Get detailed job info
    jobs_list = []
    for job_id, job in job_queue_manager.jobs.items():
        jobs_list.append({
            "job_id": job_id,
            "type": job.job_type.value,
            "status": job.status.value,
            "progress": job.progress,
            "created": job.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            "started": job.started_at.strftime('%Y-%m-%d %H:%M:%S') if job.started_at else None,
            "completed": job.completed_at.strftime('%Y-%m-%d %H:%M:%S') if job.completed_at else None,
            "error": job.error
        })
    
    return {
        "queue_status": queue_status,
        "jobs": jobs_list[-20:],  # Last 20 jobs
        "total_jobs": len(jobs_list)
    }

@app.get("/admin/files", dependencies=[Depends(verify_admin)])
async def admin_files():
    """File management"""
    files_info = []
    
    for file_id, file_path in uploaded_files.items():
        try:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                mod_time = os.path.getmtime(file_path)
                files_info.append({
                    "file_id": file_id,
                    "path": file_path,
                    "size_mb": round(file_size, 2),
                    "modified": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time)),
                    "exists": True
                })
            else:
                files_info.append({
                    "file_id": file_id,
                    "path": file_path,
                    "size_mb": 0,
                    "modified": "File not found",
                    "exists": False
                })
        except Exception as e:
            files_info.append({
                "file_id": file_id,
                "path": file_path,
                "error": str(e),
                "exists": False
            })
    
    return {
        "uploaded_files": files_info,
        "total_count": len(uploaded_files),
        "existing_count": len([f for f in files_info if f.get('exists', False)])
    }

@app.get("/admin/docs", dependencies=[Depends(verify_admin)])
async def admin_docs():
    """Protected API documentation"""
    from fastapi.openapi.docs import get_swagger_ui_html
    return get_swagger_ui_html(
        openapi_url="/admin/openapi.json",
        title="F5-TTS Admin API Documentation",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png"
    )

@app.get("/admin/openapi.json", dependencies=[Depends(verify_admin)])
async def admin_openapi():
    """Protected OpenAPI schema"""
    from fastapi.openapi.utils import get_openapi
    return get_openapi(
        title="F5-TTS API",
        version="1.0.0",
        description="REST API for F5-TTS Text-to-Speech model",
        routes=app.routes,
    )

@app.post("/admin/cleanup-jobs", dependencies=[Depends(verify_admin)])
async def admin_cleanup_jobs(max_age_hours: int = 24):
    """Cleanup old jobs (admin only)"""
    if not job_queue_manager:
        raise HTTPException(503, "Job queue not initialized")
    
    cleaned_count = job_queue_manager.cleanup_old_jobs(max_age_hours)
    return {
        "message": f"Cleaned up {cleaned_count} old jobs",
        "cleaned_count": cleaned_count
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