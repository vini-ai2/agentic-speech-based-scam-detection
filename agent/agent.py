from pathlib import Path
import importlib.util
import sys

# Prefer the real tools in the `tool-ab/tools` folder. Fall back to the
# lightweight local `tools` implementations when the heavy tool-ab code
# or model checkpoints aren't available.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOLAB_TOOLS = _REPO_ROOT / "tool-ab" / "tools"

# placeholders — we'll populate these with callables
transcribe = None
analyze_audio = None

# Load ASR tool (WhisperASR.transcribe) if present
_asr_path = _TOOLAB_TOOLS / "asr_tool.py"
if _asr_path.exists():
    spec = importlib.util.spec_from_file_location("asr_tool_module", str(_asr_path))
    _asr_mod = importlib.util.module_from_spec(spec)
    sys.modules["asr_tool_module"] = _asr_mod  # 🔥 ADD THIS
    try:
        spec.loader.exec_module(_asr_mod)
    except Exception as e:
        print("ASR LOAD ERROR:", e)
        _asr_mod = None

    if _asr_mod is not None:
        if hasattr(_asr_mod, "WhisperASR"):
            try:
                _asr_instance = _asr_mod.WhisperASR()
            except Exception:
                _asr_instance = None

            def transcribe(audio_path):
                if _asr_instance is None:
                    raise RuntimeError("ASR model failed to initialize")
                res = _asr_instance.transcribe(Path(audio_path))
                if isinstance(res, dict):
                    return res.get("text", "")
                return str(res)
        elif hasattr(_asr_mod, "transcribe"):
            transcribe = _asr_mod.transcribe

# Load acoustic tool (predict_spoof_probability) if present
_acoustic_path = _TOOLAB_TOOLS / "acoustic_tool.py"
if _acoustic_path.exists():
    spec = importlib.util.spec_from_file_location("acoustic_tool_module", str(_acoustic_path))
    _acoustic_mod = importlib.util.module_from_spec(spec)

    sys.modules["acoustic_tool_module"] = _acoustic_mod  # ✅ required

    try:
        spec.loader.exec_module(_acoustic_mod)
        print("Acoustic module loaded")

        # 🔥 ADD THIS DEBUG LINE
        print("Acoustic functions available:", dir(_acoustic_mod))

    except Exception as e:
        print("ACOUSTIC LOAD ERROR (detailed):", e)
        import traceback
        traceback.print_exc()
        _acoustic_mod = None

    if _acoustic_mod is not None:
        if hasattr(_acoustic_mod, "predict_spoof_probability"):
            def analyze_audio(audio_path):
                # Prefer one of the shipped checkpoints if available
                candidates = [
                    _REPO_ROOT / "tool-ab" / "models" / "acoustic_la_best.pt",
                    _REPO_ROOT / "tool-ab" / "models" / "acoustic_la_best_full.pt",
                    _REPO_ROOT / "tool-ab" / "models" / "acoustic_logtest.pt",
                ]
                ck = next((p for p in candidates if p.exists()), None)
                if ck is None:
                    raise FileNotFoundError("No acoustic model checkpoint found")
                prob = _acoustic_mod.predict_spoof_probability(ck, Path(audio_path))
                return {"fake_probability": float(prob), "anomaly": ""}
        elif hasattr(_acoustic_mod, "analyze_audio"):
            analyze_audio = _acoustic_mod.analyze_audio

# Fall back to local lightweight implementations if needed
if transcribe is None:
    try:
        from tools.asr import transcribe as transcribe  # type: ignore
    except Exception:
        def transcribe(audio_path):
            return ""

if analyze_audio is None:
    try:
        from tools.acoustic import analyze_audio as analyze_audio  # type: ignore
    except Exception:
        def analyze_audio(audio_path):
            return {"fake_probability": 0.5, "anomaly": "missing"}

from tools.text import analyze_text


class ScamDetectionAgent:

    def evaluate(self, audio_path):
        # Step 1: collect info
        # ASR: if it fails, default to empty transcript
        try:
            transcript = transcribe(audio_path)
            if transcript is None:
                transcript = ""
        except Exception:
            transcript = ""

        # Acoustic analysis: if it fails, provide a conservative default
        try:
            acoustic = analyze_audio(audio_path)
            if not isinstance(acoustic, dict) or "fake_probability" not in acoustic:
                acoustic = {"fake_probability": 0.5, "anomaly": "analysis_missing"}
        except Exception:
            acoustic = {"fake_probability": 0.5, "anomaly": "analysis_failed"}

        # Text analysis (works on transcript; empty string is acceptable)
        try:
            text = analyze_text(transcript)
            if not isinstance(text, dict) or "scam_score" not in text:
                text = {"scam_score": 0.0, "flags": []}
        except Exception:
            text = {"scam_score": 0.0, "flags": []}

        # Step 2: base scoring
        risk_score = (
            0.5 * acoustic["fake_probability"] +
            0.5 * text["scam_score"]
        ) * 100

        # Step 3: smarter reasoning
        # Rule 1: financial urgency keywords
        if any(word in transcript.lower() for word in ["transfer", "otp", "bank", "urgent"]):
            risk_score += 10

        # Rule 2: both signals high → very risky
        if acoustic["fake_probability"] > 0.8 and text["scam_score"] > 0.6:
            risk_score += 15

        # Rule 3: contradiction detection (cool part)
        if "urgent" in transcript.lower() and acoustic["fake_probability"] < 0.3:
            explanation_note = "Mismatch: urgent language but natural voice"
        else:
            explanation_note = "No major contradictions detected"

        # cap at 100
        risk_score = min(risk_score, 100)

        # Step 4: decision
        decision = "SCAM" if risk_score > 70 else "SAFE"

        # Step 5: explanation
        explanation = f"""
Transcript: {transcript}

Text flags: {text['flags']}
Acoustic anomaly: {acoustic['anomaly']}

Reasoning:
- Text scam score: {text['scam_score']:.2f}
- Acoustic fake probability: {acoustic['fake_probability']:.2f}
- {explanation_note}

Final Risk Score: {risk_score:.2f}
"""

        return {
            "decision": decision,
            "risk_score": round(risk_score, 2),
            "explanation": explanation.strip()
        }