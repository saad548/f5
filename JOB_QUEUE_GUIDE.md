# Job Queue System Guide

The F5-TTS API now includes a production-ready job queue system for handling concurrent users and long-running TTS generation tasks. This guide explains how to use the new asynchronous job system.

## Overview

The job queue system allows you to:
- Submit TTS generation jobs that return immediately with a job ID
- Track job progress and status in real-time
- Handle multiple concurrent users without blocking
- Process jobs sequentially to avoid GPU memory issues
- Get results when jobs complete

## Job Types

### 1. TTS Generate (`tts_generate`)
Generate TTS using an uploaded reference audio file.

### 2. TTS Permanent (`tts_permanent`) 
Generate TTS using a permanent voice from the voice bank.

### 3. Voice Cloning (`voice_cloning`)
Clone a voice and generate TTS in one operation.

## Job States

- **`queued`**: Job submitted and waiting in queue
- **`processing`**: Job currently being processed
- **`completed`**: Job finished successfully
- **`failed`**: Job encountered an error

## API Endpoints

### Generic Job Management

#### Submit Job
```http
POST /jobs/submit
```

Example request:
```json
{
  "job_type": "tts_generate",
  "parameters": {
    "ref_audio_path": "/path/to/audio.wav",
    "ref_text": "Hello world",
    "gen_text": "This is generated speech",
    "inference_settings": {
      "speed": 1.0,
      "nfe_step": 32,
      "remove_silence": false
    }
  },
  "priority": 0
}
```

Response:
```json
{
  "job_id": "abc123-def456-789",
  "message": "Job submitted successfully",
  "queue_position": 2
}
```

#### Get Job Status
```http
GET /jobs/{job_id}/status
```

Response:
```json
{
  "job_id": "abc123-def456-789",
  "job_type": "tts_generate",
  "status": "processing",
  "created_at": "2025-11-27T10:30:00",
  "started_at": "2025-11-27T10:32:00",
  "progress": 60,
  "estimated_completion": "2025-11-27T10:34:00"
}
```

#### Get Job Result
```http
GET /jobs/{job_id}/result
```

Response (when completed):
```json
{
  "output_file_id": "output_abc123",
  "output_path": "/tmp/f5tts_output_abc123.wav",
  "ref_text": "Hello world",
  "gen_text": "This is generated speech",
  "seed": 42
}
```

#### Get Queue Status
```http
GET /jobs/queue-status
```

Response:
```json
{
  "queue_size": 3,
  "total_jobs": 15,
  "job_counts": {
    "queued": 3,
    "processing": 1,
    "completed": 8,
    "failed": 3
  },
  "current_job": "abc123-def456-789"
}
```

### Convenient Async Endpoints

These endpoints are direct async versions of the existing synchronous endpoints:

#### Async TTS Generate
```http
POST /jobs/tts-generate-async
```

Form parameters:
- `audio_file_id` (required): ID of uploaded audio file
- `text` (required): Text to generate
- `ref_text` (optional): Reference text (auto-transcribed if empty)
- `randomize_seed` (default: true): Whether to randomize seed
- `seed` (default: 0): Random seed for generation
- `remove_silence` (default: false): Remove silence from output
- `speed` (default: 1.0): Speech speed (0.3-2.0)
- `nfe_step` (default: 32): Number of inference steps (4-64)
- `cross_fade_duration` (default: 0.15): Cross-fade duration (0.0-1.0)
- `priority` (default: 0): Job priority (higher = more priority)

#### Async Permanent Voice TTS
```http
POST /jobs/tts-permanent-async
```

Form parameters:
- `voice_name` (required): Name of permanent voice
- `text` (required): Text to generate
- Plus all the same inference settings as above

#### Async Voice Cloning
```http
POST /jobs/voice-cloning-async
```

Form parameters:
- `audio_file_id` (required): ID of uploaded audio file
- `text` (required): Text to generate
- Plus all the same inference settings as above

## Usage Examples

### Python Client Example

```python
import requests
import time

# Submit a TTS job
response = requests.post("http://localhost:8000/jobs/tts-generate-async", 
    data={
        "audio_file_id": "your_audio_id",
        "text": "Hello, this is a test of the job queue system!",
        "speed": 1.2,
        "priority": 1
    }
)
job_data = response.json()
job_id = job_data["job_id"]
print(f"Job submitted: {job_id}")

# Poll for completion
while True:
    status_response = requests.get(f"http://localhost:8000/jobs/{job_id}/status")
    status = status_response.json()
    
    print(f"Status: {status['status']}, Progress: {status['progress']}%")
    
    if status['status'] == 'completed':
        # Get the result
        result_response = requests.get(f"http://localhost:8000/jobs/{job_id}/result")
        result = result_response.json()
        output_file_id = result['output_file_id']
        
        # Download the generated audio
        audio_response = requests.get(f"http://localhost:8000/download-audio/{output_file_id}")
        with open("generated_speech.wav", "wb") as f:
            f.write(audio_response.content)
        
        print("Audio generated and saved!")
        break
    
    elif status['status'] == 'failed':
        print(f"Job failed: {status.get('error', 'Unknown error')}")
        break
    
    time.sleep(2)  # Wait 2 seconds before checking again
```

### JavaScript/Fetch Example

```javascript
async function generateSpeechAsync(audioFileId, text) {
    // Submit job
    const submitResponse = await fetch('/jobs/tts-generate-async', {
        method: 'POST',
        body: new FormData(Object.entries({
            audio_file_id: audioFileId,
            text: text,
            speed: '1.0',
            priority: '0'
        }).reduce((fd, [k, v]) => (fd.append(k, v), fd), new FormData()))
    });
    
    const submitData = await submitResponse.json();
    const jobId = submitData.job_id;
    console.log(`Job submitted: ${jobId}`);
    
    // Poll for completion
    while (true) {
        const statusResponse = await fetch(`/jobs/${jobId}/status`);
        const status = await statusResponse.json();
        
        console.log(`Status: ${status.status}, Progress: ${status.progress}%`);
        
        if (status.status === 'completed') {
            // Get result
            const resultResponse = await fetch(`/jobs/${jobId}/result`);
            const result = await resultResponse.json();
            
            // Create download link
            const audioUrl = `/download-audio/${result.output_file_id}`;
            console.log('Audio ready:', audioUrl);
            return audioUrl;
        }
        
        if (status.status === 'failed') {
            throw new Error(`Job failed: ${status.error}`);
        }
        
        await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds
    }
}
```

## Job Management

### Cleanup Old Jobs
```http
POST /jobs/cleanup?max_age_hours=24
```

Remove completed/failed jobs older than specified hours to free memory.

### Monitor Queue
Check `/jobs/queue-status` regularly to monitor system load and performance.

## Performance Notes

1. **Sequential Processing**: Only one TTS job processes at a time to avoid GPU memory issues
2. **In-Memory Storage**: Job data is stored in memory - jobs are lost on server restart
3. **File Management**: Generated audio files are stored temporarily and managed automatically
4. **Priority Support**: Higher priority jobs are processed first (useful for premium users)

## Migration from Synchronous API

The original synchronous endpoints (`/tts-generate`, `/tts-permanent`, `/voice-cloning`) still work exactly as before. The new async endpoints provide the same functionality but return immediately with a job ID instead of blocking until completion.

For production deployments with multiple users, we recommend using the async endpoints to provide a better user experience.

## Error Handling

Jobs can fail for various reasons:
- Invalid audio files
- Model loading errors
- GPU memory issues
- Invalid parameters

Always check the job status and handle the `failed` state appropriately in your client code.

## Scaling Considerations

The current implementation uses a single background worker thread. For higher throughput:
1. The `max_workers` parameter can be increased if you have multiple GPUs
2. Consider implementing Redis-based job queue for multi-server setups
3. Add job persistence for server restart resilience

The in-memory approach is perfect for single-server deployments and provides excellent performance for most use cases.