# 🎤 Bulk Voice Upload Guide

## 📁 Directory Structure

Organize your files like this:

```
ff5 - tts/
├── voice_samples/           # Put your 26 MP3 files here
│   ├── Adam.mp3
│   ├── Alice.mp3
│   ├── Bill.mp3
│   └── ... (23 more)
│
├── voice_metadata/          # Put your JSON files here
│   ├── Adam.json
│   ├── Alice.json
│   ├── Bill.json
│   └── ... (23 more)
│
└── upload_voices_bulk.py    # Upload script
```

## 📊 Metadata Format

Each JSON file should contain:

```json
{
  "name": "Daniel",
  "voice_id": "onwK4e9ZLuTAKqWW03F9",
  "category": "premade",
  "labels": {
    "accent": "british",
    "descriptive": "formal",
    "age": "middle_aged",
    "gender": "male",
    "language": "en",
    "use_case": "informative_educational"  // ← NEW: informative_educational, conversational, narrative, etc.
  },
  "description": "A strong voice perfect for delivering a professional broadcast or news story."
}
```

## 🚀 Usage

### Step 1: Organize Files
```bash
# Create directories
mkdir voice_samples voice_metadata

# Copy your 26 MP3 files to voice_samples/
# Copy your 26 JSON files to voice_metadata/
```

### Step 2: Run Upload Script
```bash
python upload_voices_bulk.py
```

### Step 3: Verify Upload
The script will automatically:
- ✅ Upload all 26 voices
- ✅ Upload all metadata
- ✅ Test listing with filters
- ✅ Show summary

## 📡 API Endpoints Added

### 1. Bulk Upload
```bash
POST /bulk-upload-voices
Content-Type: multipart/form-data

# Upload multiple files at once
# With optional metadata JSON
```

### 2. List Voices with Filters
```bash
# Get all voices
GET /list-voices

# Filter by gender
GET /list-voices?gender=male

# Filter by age
GET /list-voices?age=middle_aged

# Filter by accent
GET /list-voices?accent=british

# Filter by category
GET /list-voices?category=premade

# Filter by use_case (NEW!)
GET /list-voices?use_case=informative_educational

# Combine filters
GET /list-voices?gender=female&age=young&use_case=conversational
```

## 📋 Response Format

### List Voices Response:
```json
{
  "total": 26,
  "voices": [
    {
      "voice_name": "Daniel",
      "filename": "Daniel.mp3",
      "size": "0.5 MB",
      "duration": 5.2,
      "path": "/path/to/Daniel.mp3",
      "use_case": "informative_educational",
      "metadata": {
        "name": "Daniel",
        "category": "premade",
        "labels": {
          "accent": "british",
          "age": "middle_aged",
          "gender": "male",
          "use_case": "informative_educational"
        },
        "description": "Professional broadcast voice"
      }
    }
  ]
}
```

### Bulk Upload Response:
```json
{
  "message": "Bulk upload completed: 26 successful, 0 failed",
  "uploaded_count": 26,
  "voices": [
    {
      "name": "Daniel",
      "file_name": "Daniel.mp3",
      "duration": 5.2,
      "metadata": { ... }
    }
  ],
  "failed": []
}
```

## 🎯 Frontend Integration

### Filter Voices by Gender:
```javascript
const response = await fetch(`${API_URL}/list-voices?gender=male`, {
    headers: { 'X-API-Key': API_KEY }
});
const data = await response.json();

// data.voices contains only male voices
data.voices.forEach(voice => {
    console.log(voice.voice_name, voice.metadata.labels.age);
});
```

### Display Voice Cards:
```javascript
voices.forEach(voice => {
    const metadata = voice.metadata;
    const labels = metadata?.labels || {};
    
    // Create voice card
    const card = `
        <div class="voice-card">
            <h3>🎤 ${voice.voice_name}</h3>
            <p>👤 ${labels.gender || 'N/A'}</p>
            <p>🎂 ${labels.age || 'N/A'}</p>
            <p>🌍 ${labels.accent || 'N/A'}</p>
            <p>🎯 Use: ${voice.use_case || labels.use_case || 'N/A'}</p>
            <p>⏱️ ${voice.duration}s</p>
            <button onclick="selectVoice('${voice.voice_name}')">
                Select
            </button>
        </div>
    `;
});
```

## ✅ Benefits

1. **Upload Once**: All 26 voices uploaded in one request
2. **Rich Metadata**: Gender, age, accent, category, use_case, description
3. **Smart Filtering**: Filter by any metadata field (including use_case)
4. **Professional UX**: Display detailed voice info
5. **Easy Management**: Metadata stored as JSON files

## 🎨 Voice Discovery

Your frontend can now:
- 🔍 Filter voices by characteristics (gender, age, accent, use_case)
- 📊 Show voice statistics
- 🎯 Recommend voices based on use case (informative, conversational, narrative)
- 🏷️ Tag and categorize voices
- 📈 Sort by duration, accent, use_case, etc.

Perfect for a professional voice selection UI! 🚀
