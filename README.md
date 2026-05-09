# Agentic Speech-Based Misinformation and Scam Call Detection 
## Problem explanation 
Detect scam and misinformation speech (fraud calls, fake announcements) by jointly 
analyzing acoustic cues, speaking style, and ASR-derived semantics, coordinated by an 
intelligent agent. 

### Why this matters in 2025 
Voice-based scams and misinformation are growing rapidly, especially with AI-generated 
voices.

### Dataset (2024–2025 only) 
• ASVspoof 2024–25 (logical + real spoof) 
• Common Voice 2024–25 (for benign speech) 

### Key limitation in existing work 
• Text-only scam detection 
• Ignores speech delivery patterns 

### New research contribution 
• Multi-modal speech + text analysis 
• Agent fuses acoustic risk + semantic risk 
• Explainable risk scoring 

## 2025 references 
• Zhang, Audio Deepfake Detection: Current State and Open Problems, Sensors 
2025 
• INTERSPEECH 2025: Speech-Based Fraud and Scam Detection

## Run the agent

### 1) Set up Python
Use a virtual environment (recommended) or your system Python. Example with conda:

```bash
conda create -n scam_env python=3.10
conda activate scam_env
```

### 2) Install dependencies
Install the agent tool dependencies:

```bash
pip install -r tool-ab/requirements.txt
```

Note: MP3 decoding may require ffmpeg on some systems. If you see audio decode errors, try a WAV file or install ffmpeg.
When ffmpeg is available, MP3 files are automatically converted to WAV for ASR.

Note: If you hit PyTorch DLL errors on Windows, reinstall torch/torchaudio for your CPU:

```bash
pip uninstall -y torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Note: Whisper models are downloaded from Hugging Face on first use. To ensure stable offline operation, pre-download and cache the models:

```bash
cd tool-ab
python download_whisper_models.py --model openai/whisper-small --cache-dir ./models/whisper_cache
```

Then set the environment variable to use the cached models:

```bash
setx AGENTIC_ASR_CACHE_DIR "path/to/tool-ab/models/whisper_cache"
```

Or add to .env at the repo root:

```
AGENTIC_ASR_CACHE_DIR=tool-ab/models/whisper_cache
```

This ensures the ASR model loads from local cache instead of hitting Hugging Face on every run, avoiding transient network failures.

Optional: LLM explainability (agentic mode). Set an OpenAI-compatible endpoint:

```bash
setx AGENTIC_LLM_API_KEY "your_api_key"
setx AGENTIC_LLM_MODEL "gpt-4o-mini"
setx AGENTIC_LLM_BASE_URL "https://api.openai.com/v1"
```

Or add the same keys to a repo-root .env file (auto-loaded by the agent).

Optional ASR tuning (add to .env):

```bash
AGENTIC_ASR_LANGUAGE=en
AGENTIC_ASR_CHUNK_SECONDS=30
AGENTIC_ASR_MODEL=openai/whisper-small
```

Groq example (OpenAI-compatible):

```bash
setx GROQ_API_KEY "your_groq_key"
setx AGENTIC_LLM_PROVIDER "groq"
setx AGENTIC_LLM_MODEL "llama3-8b-8192"
```

To disable LLM explainability for a run:

```bash
python -m agent.main path/to/audio.wav --no-llm
```

### 3) Run the demo agent
From the repo root:

```bash
python -m agent.main path/to/audio.wav
```

Or from inside the folder:

```bash
cd agent
python main.py path/to/audio.wav
```

### 4) Use the agent in code
Example usage from Python:

```python
from agent import AgenticAIScamDetectionAgent

agent = AgenticAIScamDetectionAgent()
result = agent.evaluate("path/to/audio.wav")
print(result)
```