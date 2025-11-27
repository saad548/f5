# 🎭 F5-TTS Voice Management & Cloning Guide

## 🎯 **Two TTS Systems:**

### 1. **Permanent Voice Bank** 📚
- Upload voices with readable names like "adam", "sarah", "narrator"
- Voices stored permanently in `reference_voices/` folder
- Use for consistent characters, brands, etc.

### 2. **Voice Cloning** 🎪 
- Upload + generate + delete in one call
- Perfect for one-time voice cloning
- No permanent storage

---

## 📁 **Permanent Voice System**

### **Upload Permanent Voice:**
```bash
curl -X POST "http://localhost:8000/upload-permanent-voice" \
     -F "voice_name=adam" \
     -F "audio_file=@adam_voice.wav"
```

### **Generate with Permanent Voice:**
```bash
curl -X POST "http://localhost:8000/tts-permanent" \
     -H "Content-Type: application/json" \
     -d '{
       "voice_name": "adam",
       "text": "Hello, this is Adam speaking!",
       "settings": {
         "speed": 1.1,
         "nfe_step": 32
       }
     }'
```

### **List All Voices:**
```bash
curl "http://localhost:8000/list-voices"
```

**Response:**
```json
{
  "voices": [
    {"name": "adam", "filename": "adam.wav", "duration": 8.5},
    {"name": "sarah", "filename": "sarah.mp3", "duration": 6.2},
    {"name": "narrator", "filename": "narrator.wav", "duration": 12.0}
  ],
  "total_voices": 3,
  "storage_path": "reference_voices"
}
```

---

## 🎪 **Voice Cloning (One-Shot)**

### **Clone Voice + Generate (All-in-One):**
```bash
curl -X POST "http://localhost:8000/voice-cloning" \
     -F "text=This is a voice cloning test!" \
     -F "ref_text=" \
     -F "audio_file=@some_voice.wav" \
     -F "speed=1.2" \
     -F "remove_silence=true"
```

**What happens:**
1. ✅ Uploads audio temporarily
2. ✅ Auto-transcribes reference text
3. ✅ Generates TTS audio
4. ✅ **Deletes reference audio immediately**
5. ✅ Returns generated audio file ID

---

## 🗂️ **File Organization**

### **Directory Structure:**
```
project/
├── reference_voices/          # Permanent voice bank
│   ├── adam.wav              # Readable names
│   ├── sarah.mp3
│   └── narrator.wav
├── /tmp/                     # Temporary files
│   ├── f5tts_ref_uuid.wav    # Temp references
│   ├── f5tts_output_uuid.wav # Generated audio
│   └── f5tts_clone_uuid.wav  # Cloning temp files (auto-deleted)
└── f5_tts_api.py
```

---

## 🎨 **Use Cases**

### **Permanent Voices:** 
- ✅ **Character voices** for games/stories
- ✅ **Brand voices** for consistent marketing
- ✅ **Narrator voices** for content creation
- ✅ **Personal voice library**

### **Voice Cloning:**
- ✅ **One-time mimicking**
- ✅ **Privacy-conscious cloning** (no storage)
- ✅ **Quick experiments**
- ✅ **Demo purposes**

---

## 📋 **All Available Endpoints:**

| Endpoint | Purpose | Storage |
|----------|---------|---------|
| `/upload-audio` | Temporary reference | Temp |
| `/upload-permanent-voice` | Permanent voice bank | Permanent |
| `/tts-generate` | Use temp reference | - |
| `/tts-permanent` | Use permanent voice | - |
| `/voice-cloning` | Upload + generate + delete | None |
| `/list-voices` | Show voice bank | - |

Perfect for both professional voice management and quick voice cloning! 🎵