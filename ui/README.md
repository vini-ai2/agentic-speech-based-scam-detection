Agentic Scam Detector — Flask UI

Quick start

1. Create a virtualenv and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements-ui.txt
```

2. Run the app:

```bash
python -m ui.app
```

3. Open http://localhost:8501

Notes
- The UI saves uploaded files to `ui/uploads` and writes analysis JSON there.
- To enable LLM explainability set `AGENTIC_LLM_API_KEY` in your environment.
