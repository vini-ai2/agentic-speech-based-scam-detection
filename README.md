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
pip install librosa
```

Note: If you hit PyTorch DLL errors on Windows, reinstall torch/torchaudio for your CPU:

```bash
pip uninstall -y torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3) Run the demo agent
From the repo root:

```bash
cd agent
python main.py
```

### 4) Use the agent in code
Example usage from Python:

```python
from agent import ScamDetectionAgent

agent = ScamDetectionAgent()
result = agent.evaluate("path/to/audio.wav")
print(result)
```