#!/usr/bin/env python3
"""
Bulk Upload Script for F5-TTS Voices with Metadata
Usage: python upload_voices_bulk.py
"""

import requests
import json
import os
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"  # Change to your server URL
API_KEY = "speechora_f5tts_api_key_2025_secure_xyz789"

VOICE_SAMPLES_DIR = "voice_samples"  # Directory with your MP3 files
METADATA_DIR = "voice_metadata"  # Directory with your JSON files

def bulk_upload():
    """Upload all voices with metadata to API."""
    
    # Check if directories exist
    if not os.path.exists(VOICE_SAMPLES_DIR):
        print(f"❌ Error: {VOICE_SAMPLES_DIR} directory not found")
        return
    
    # Collect all voice files
    voice_files = []
    for ext in ['*.mp3', '*.wav']:
        voice_files.extend(Path(VOICE_SAMPLES_DIR).glob(ext))
    
    if not voice_files:
        print(f"❌ No voice files found in {VOICE_SAMPLES_DIR}")
        return
    
    print(f"📁 Found {len(voice_files)} voice files")
    
    # Load all metadata into single JSON
    all_metadata = {}
    if os.path.exists(METADATA_DIR):
        for metadata_file in Path(METADATA_DIR).glob('*.json'):
            if metadata_file.name == 'all_voices.json':
                continue
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    voice_name = metadata_file.stem  # Get filename without extension
                    all_metadata[voice_name] = data
                    print(f"✅ Loaded metadata for: {voice_name}")
            except Exception as e:
                print(f"⚠️  Error loading {metadata_file.name}: {e}")
    
    # Save combined metadata
    combined_metadata_path = "all_voices_combined.json"
    with open(combined_metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    print(f"\n💾 Created combined metadata file: {combined_metadata_path}")
    
    # Prepare multipart form data
    files = []
    
    # Add all voice files
    for voice_file in voice_files:
        files.append(
            ('voice_files', (voice_file.name, open(voice_file, 'rb'), 'audio/mpeg'))
        )
    
    # Add metadata file
    files.append(
        ('metadata_file', ('all_voices.json', open(combined_metadata_path, 'rb'), 'application/json'))
    )
    
    # Upload to API
    print(f"\n🚀 Uploading to {API_URL}/bulk-upload-voices...")
    try:
        response = requests.post(
            f"{API_URL}/bulk-upload-voices",
            files=files,
            headers={"X-API-Key": API_KEY}
        )
        
        # Close all file handles
        for _, file_tuple in files:
            file_tuple[1].close()
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ SUCCESS!")
            print(f"📊 Uploaded: {result['uploaded_count']} voices")
            print(f"\n📋 Uploaded voices:")
            for voice in result['voices']:
                metadata = voice.get('metadata', {})
                labels = metadata.get('labels', {})
                print(f"  🎤 {voice['name']}")
                print(f"     Gender: {labels.get('gender', 'N/A')}")
                print(f"     Age: {labels.get('age', 'N/A')}")
                print(f"     Accent: {labels.get('accent', 'N/A')}")
                print(f"     Duration: {voice.get('duration', 'N/A')}s")
                print()
            
            if result.get('failed'):
                print(f"\n⚠️  Failed uploads: {len(result['failed'])}")
                for failed in result['failed']:
                    print(f"  ❌ {failed['file']}: {failed['error']}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Upload failed: {e}")

def test_list_voices():
    """Test listing voices with filters."""
    print("\n🔍 Testing voice listing...")
    
    try:
        # List all voices
        response = requests.get(
            f"{API_URL}/list-voices",
            headers={"X-API-Key": API_KEY}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n📊 Total voices: {result['total']}")
            
            # Test filters
            print("\n🔍 Testing filters...")
            
            # Filter by gender
            response = requests.get(
                f"{API_URL}/list-voices?gender=male",
                headers={"X-API-Key": API_KEY}
            )
            if response.status_code == 200:
                male_voices = response.json()
                print(f"  👨 Male voices: {male_voices['total']}")
            
            # Filter by age
            response = requests.get(
                f"{API_URL}/list-voices?age=middle_aged",
                headers={"X-API-Key": API_KEY}
            )
            if response.status_code == 200:
                middle_aged = response.json()
                print(f"  🎂 Middle-aged voices: {middle_aged['total']}")
            
            # Filter by accent
            response = requests.get(
                f"{API_URL}/list-voices?accent=british",
                headers={"X-API-Key": API_KEY}
            )
            if response.status_code == 200:
                british = response.json()
                print(f"  🇬🇧 British accent: {british['total']}")
                
    except Exception as e:
        print(f"❌ List test failed: {e}")

if __name__ == "__main__":
    print("🎵 F5-TTS Bulk Voice Upload Tool")
    print("=" * 50)
    
    bulk_upload()
    test_list_voices()
    
    print("\n✅ Done!")
