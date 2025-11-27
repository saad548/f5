#!/usr/bin/env python3
"""
Test script for F5-TTS Job Queue System
Tests the async job submission and monitoring functionality.
"""

import requests
import time
import json
from pathlib import Path

# Configuration
API_BASE = "http://213.181.123.57:20679/"
TEST_TEXT = "Hello! This is a test of the job queue system. The quick brown fox jumps over the lazy dog."

def test_job_queue_system():
    """Test the complete job queue workflow"""
    
    print("🚀 Testing F5-TTS Job Queue System")
    print("=" * 50)
    
    # 1. Check if server is running
    print("1. Checking server status...")
    try:
        response = requests.get(f"{API_BASE}/")
        print(f"✅ Server is running: {response.json()['message']}")
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running. Please start the F5-TTS API server first.")
        return
    
    # 2. Check queue status
    print("\n2. Checking initial queue status...")
    try:
        response = requests.get(f"{API_BASE}/jobs/queue-status")
        queue_status = response.json()
        print(f"✅ Queue status: {queue_status}")
    except Exception as e:
        print(f"❌ Queue status check failed: {e}")
        return
    
    # 3. List available permanent voices
    print("\n3. Checking available permanent voices...")
    try:
        response = requests.get(f"{API_BASE}/list-voices")
        voices = response.json()
        print(f"✅ Found {voices['total_voices']} permanent voices")
        
        if voices['total_voices'] == 0:
            print("⚠️  No permanent voices found. You'll need to upload some voices first.")
            print("   Use /upload-permanent-voice endpoint to add voices.")
            return
        
        # Print all available voices for debugging
        print("   Available voices:")
        for voice in voices['voices']:
            print(f"   - {voice['name']} ({voice['filename']})")
        
        # Use the first available voice for testing
        test_voice = voices['voices'][0]['name']
        print(f"📢 Using voice '{test_voice}' for testing")
        
    except Exception as e:
        print(f"❌ Voice listing failed: {e}")
        return
    
    # 4. Submit async TTS job
    print(f"\n4. Submitting async TTS job with voice '{test_voice}'...")
    try:
        job_data = {
            'voice_name': test_voice,
            'text': TEST_TEXT,
            'speed': '1.1',
            'nfe_step': '16',  # Faster for testing
            'priority': '1'
        }
        
        response = requests.post(f"{API_BASE}/jobs/tts-permanent-async", data=job_data)
        
        if response.status_code != 200:
            print(f"❌ Job submission failed: {response.text}")
            return
            
        submit_result = response.json()
        job_id = submit_result['job_id']
        print(f"✅ Job submitted successfully!")
        print(f"   Job ID: {job_id}")
        print(f"   Queue position: {submit_result['queue_position']}")
        
    except Exception as e:
        print(f"❌ Job submission failed: {e}")
        return
    
    # 5. Monitor job progress
    print(f"\n5. Monitoring job progress...")
    start_time = time.time()
    last_progress = -1
    
    while True:
        try:
            # Check job status
            response = requests.get(f"{API_BASE}/jobs/{job_id}/status")
            status = response.json()
            
            current_progress = status.get('progress', 0)
            current_status = status['status']
            
            # Only print if progress changed or status changed
            if current_progress != last_progress:
                elapsed = time.time() - start_time
                print(f"   📊 Status: {current_status} | Progress: {current_progress}% | Elapsed: {elapsed:.1f}s")
                last_progress = current_progress
            
            if current_status == 'completed':
                print(f"✅ Job completed successfully in {elapsed:.1f} seconds!")
                break
            elif current_status == 'failed':
                error_msg = status.get('error', 'Unknown error')
                print(f"❌ Job failed: {error_msg}")
                return
            elif current_status in ['queued', 'processing']:
                time.sleep(1)  # Wait 1 second before next check
            else:
                print(f"❓ Unknown status: {current_status}")
                break
                
        except Exception as e:
            print(f"❌ Status check failed: {e}")
            return
        
        # Timeout after 5 minutes
        if time.time() - start_time > 300:
            print("⏰ Job timed out after 5 minutes")
            return
    
    # 6. Get job result
    print("\n6. Getting job result...")
    try:
        response = requests.get(f"{API_BASE}/jobs/{job_id}/result")
        result = response.json()
        
        output_file_id = result['output_file_id']
        print(f"✅ Result retrieved successfully!")
        print(f"   Output file ID: {output_file_id}")
        print(f"   Reference text: {result['ref_text'][:50]}...")
        print(f"   Generated text: {result['gen_text'][:50]}...")
        print(f"   Seed used: {result['seed']}")
        
    except Exception as e:
        print(f"❌ Result retrieval failed: {e}")
        return
    
    # 7. Download generated audio
    print("\n7. Downloading generated audio...")
    try:
        response = requests.get(f"{API_BASE}/download-audio/{output_file_id}")
        
        if response.status_code == 200:
            output_file = f"test_job_queue_output_{int(time.time())}.wav"
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            file_size = len(response.content)
            print(f"✅ Audio downloaded successfully!")
            print(f"   File: {output_file}")
            print(f"   Size: {file_size:,} bytes")
            
        else:
            print(f"❌ Audio download failed: {response.status_code} - {response.text}")
            return
            
    except Exception as e:
        print(f"❌ Audio download failed: {e}")
        return
    
    # 8. Final queue status
    print("\n8. Final queue status...")
    try:
        response = requests.get(f"{API_BASE}/jobs/queue-status")
        final_status = response.json()
        print(f"✅ Final queue status: {final_status}")
    except Exception as e:
        print(f"❌ Final status check failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Job queue test completed successfully!")
    print(f"📁 Generated audio saved as: {output_file}")
    print("🔧 The async job system is working correctly!")

def test_multiple_jobs():
    """Test submitting multiple jobs to verify queue handling"""
    
    print("\n🔄 Testing multiple job submission...")
    
    # Get available voice
    try:
        response = requests.get(f"{API_BASE}/list-voices")
        voices = response.json()
        if voices['total_voices'] == 0:
            print("⚠️  No permanent voices available for multi-job test")
            return
        test_voice = voices['voices'][0]['name']
    except Exception as e:
        print(f"❌ Failed to get voices: {e}")
        return
    
    # Submit 3 jobs quickly
    job_ids = []
    for i in range(3):
        try:
            job_data = {
                'voice_name': test_voice,
                'text': f"This is test job number {i+1}. Testing concurrent job handling.",
                'nfe_step': '8',  # Very fast for testing
                'priority': str(i)  # Different priorities
            }
            
            response = requests.post(f"{API_BASE}/jobs/tts-permanent-async", data=job_data)
            result = response.json()
            job_ids.append(result['job_id'])
            print(f"✅ Job {i+1} submitted: {result['job_id']}")
            
        except Exception as e:
            print(f"❌ Job {i+1} submission failed: {e}")
    
    # Monitor all jobs
    if job_ids:
        print(f"\n📊 Monitoring {len(job_ids)} jobs...")
        completed = set()
        
        while len(completed) < len(job_ids):
            for job_id in job_ids:
                if job_id not in completed:
                    try:
                        response = requests.get(f"{API_BASE}/jobs/{job_id}/status")
                        status = response.json()
                        
                        if status['status'] == 'completed':
                            completed.add(job_id)
                            print(f"✅ Job {job_id[:8]} completed")
                        elif status['status'] == 'failed':
                            completed.add(job_id)
                            print(f"❌ Job {job_id[:8]} failed")
                    except:
                        pass
            
            if len(completed) < len(job_ids):
                time.sleep(1)
        
        print(f"🎉 All {len(job_ids)} jobs processed!")

if __name__ == "__main__":
    print("F5-TTS Job Queue Test Suite")
    print("=" * 40)
    
    test_job_queue_system()
    
    # Uncomment to test multiple jobs
    # test_multiple_jobs()
    
    print("\n🏁 Test suite completed!")