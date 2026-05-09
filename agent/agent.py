from pathlib import Path
import importlib.util
import json
import os
import sys
import urllib.error
import urllib.request

# Prefer the real tools in the `tool-ab/tools` folder. Fall back to the
# lightweight local `tools` implementations when the heavy tool-ab code
# or model checkpoints aren't available.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOLAB_TOOLS = _REPO_ROOT / "tool-ab" / "tools"


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    try:
        content = env_path.read_text(encoding="utf-8")
    except Exception:
        return
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(_REPO_ROOT / ".env")

# placeholders — we'll populate these with callables
transcribe = None
analyze_audio = None
TRANSCRIBE_SOURCE = "unavailable"
ANALYZE_SOURCE = "unavailable"

# Load ASR tool (WhisperASR.transcribe) if present
_asr_path = _TOOLAB_TOOLS / "asr_tool.py"
if _asr_path.exists():
    spec = importlib.util.spec_from_file_location("asr_tool_module", str(_asr_path))
    _asr_mod = importlib.util.module_from_spec(spec)
    sys.modules["asr_tool_module"] = _asr_mod
    try:
        spec.loader.exec_module(_asr_mod)
    except Exception as e:
        print("ASR LOAD ERROR:", e)
        _asr_mod = None

    if _asr_mod is not None:
        if hasattr(_asr_mod, "WhisperASR"):
            try:
                model_name = os.getenv("AGENTIC_ASR_MODEL") or "openai/whisper-small"
                cache_dir = os.getenv("AGENTIC_ASR_CACHE_DIR")
                _asr_instance = _asr_mod.WhisperASR(model_name=model_name, cache_dir=cache_dir)
            except Exception as e:
                print(f"ASR initialization error: {e}")
                _asr_instance = None

            def transcribe(audio_path):
                if _asr_instance is None:
                    raise RuntimeError("ASR model failed to initialize")
                language = os.getenv("AGENTIC_ASR_LANGUAGE") or None
                res = _asr_instance.transcribe(Path(audio_path), language=language)
                if isinstance(res, dict):
                    return res.get("text", "")
                return str(res)
            TRANSCRIBE_SOURCE = "tool-ab"
        elif hasattr(_asr_mod, "transcribe"):
            transcribe = _asr_mod.transcribe
            TRANSCRIBE_SOURCE = "tool-ab"

# Load acoustic tool (predict_spoof_probability) if present
_acoustic_path = _TOOLAB_TOOLS / "acoustic_tool.py"
if _acoustic_path.exists():
    spec = importlib.util.spec_from_file_location("acoustic_tool_module", str(_acoustic_path))
    _acoustic_mod = importlib.util.module_from_spec(spec)
    sys.modules["acoustic_tool_module"] = _acoustic_mod

    try:
        spec.loader.exec_module(_acoustic_mod)

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
                result = _acoustic_mod.predict_spoof_probability(ck, Path(audio_path))
                if isinstance(result, dict):
                    fake_probability = float(result.get("fake_probability", 0.5))
                    raw_fake_probability = float(result.get("raw_fake_probability", fake_probability))
                    anomaly = str(result.get("anomaly", ""))
                else:
                    fake_probability = float(result)
                    raw_fake_probability = fake_probability
                    anomaly = ""
                return {
                    "fake_probability": fake_probability,
                    "raw_fake_probability": raw_fake_probability,
                    "anomaly": anomaly,
                }
            ANALYZE_SOURCE = "tool-ab"
        elif hasattr(_acoustic_mod, "analyze_audio"):
            analyze_audio = _acoustic_mod.analyze_audio
            ANALYZE_SOURCE = "tool-ab"

# Fall back to local lightweight implementations if needed
if transcribe is None:
    try:
        from .tools.asr import transcribe as transcribe  # type: ignore
        TRANSCRIBE_SOURCE = "stub"
    except Exception:
        try:
            from tools.asr import transcribe as transcribe  # type: ignore
            TRANSCRIBE_SOURCE = "stub"
        except Exception:
            def transcribe(audio_path):
                return ""
            TRANSCRIBE_SOURCE = "stub"

if analyze_audio is None:
    try:
        from .tools.acoustic import analyze_audio as analyze_audio  # type: ignore
        ANALYZE_SOURCE = "stub"
    except Exception:
        try:
            from tools.acoustic import analyze_audio as analyze_audio  # type: ignore
            ANALYZE_SOURCE = "stub"
        except Exception:
            def analyze_audio(audio_path):
                return {"fake_probability": 0.5, "anomaly": "missing"}
            ANALYZE_SOURCE = "stub"

try:
    from .tools.text import analyze_text
except Exception:
    from tools.text import analyze_text


def _truncate_text(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


class LLMExplainabilityAgent:
    def __init__(self) -> None:
        provider = os.getenv("AGENTIC_LLM_PROVIDER", "").strip().lower()
        groq_key = os.getenv("GROQ_API_KEY")
        self.api_key = os.getenv("AGENTIC_LLM_API_KEY") or groq_key
        self.last_error = None
        if provider == "groq" or (provider == "" and groq_key):
            self.base_url = os.getenv("AGENTIC_LLM_BASE_URL", "https://api.groq.com/openai/v1")
            self.model = os.getenv("AGENTIC_LLM_MODEL", os.getenv("GROQ_MODEL", "llama3-8b-8192"))
        else:
            self.base_url = os.getenv("AGENTIC_LLM_BASE_URL", "https://api.openai.com/v1")
            self.model = os.getenv("AGENTIC_LLM_MODEL", "gpt-4o-mini")
        self.timeout = int(os.getenv("AGENTIC_LLM_TIMEOUT", "25"))
        self.enabled = bool(self.api_key)

    def explain(
        self,
        transcript: str,
        text_flags: list,
        text_score: float,
        acoustic_prob: float,
        acoustic_anomaly: str,
        decision: str,
        risk_score: float,
    ) -> str | None:
        if not self.enabled:
            return None

        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an explainability agent for a scam call detector. "
                        "Use only the provided evidence. Do not change the decision or risk score. "
                        "Explain in short bullet points, highlight uncertainties and contradictions, "
                        "and keep the response under 1200 characters."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "decision": decision,
                            "risk_score": round(risk_score, 2),
                            "transcript": _truncate_text(transcript),
                            "text_flags": text_flags,
                            "text_score": round(text_score, 4),
                            "acoustic_fake_probability": round(acoustic_prob, 4),
                            "acoustic_anomaly": acoustic_anomaly,
                        },
                        ensure_ascii=True,
                        indent=2,
                    ),
                },
            ],
        }

        url = self.base_url.rstrip("/") + "/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "agentic-scam-detection/1.0",
        }
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
            parsed = json.loads(body)
            content = parsed["choices"][0]["message"]["content"].strip()
            self.last_error = None
            return content
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = ""
            self.last_error = f"HTTP {exc.code} {exc.reason} {detail}".strip()
            return None
        except urllib.error.URLError as exc:
            self.last_error = f"URL error: {exc.reason}".strip()
            return None
        except (KeyError, IndexError, ValueError) as exc:
            self.last_error = f"Parse error: {exc}".strip()
            return None
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError, ValueError):
            return None


class ScamDetectionAgent:
    def __init__(self, use_llm: bool = True) -> None:
        self.use_llm = use_llm
        self._llm_agent = LLMExplainabilityAgent() if use_llm else None

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

        # Step 2: dynamic fusion
        asr_present = bool(transcript and transcript.strip())
        if not asr_present:
            w_acoustic = 0.8
            w_text = 0.2
        else:
            w_acoustic = 0.5
            w_text = 0.5

        risk_score = (w_acoustic * acoustic["fake_probability"] + w_text * text["scam_score"]) * 100

        # Step 3: smarter reasoning
        if any(word in transcript.lower() for word in ["transfer", "otp", "bank", "urgent"]):
            risk_score += 10

        if acoustic["fake_probability"] > 0.8 and text["scam_score"] > 0.6:
            risk_score += 15

        if not asr_present and acoustic["fake_probability"] >= 0.85:
            risk_score += 20

        if acoustic["fake_probability"] >= 0.95:
            risk_score = max(risk_score, 90)

        if "urgent" in transcript.lower() and acoustic["fake_probability"] < 0.3:
            explanation_note = "Mismatch: urgent language but natural voice"
        else:
            explanation_note = "No major contradictions detected"

        risk_score = min(risk_score, 100)

        # Step 4: decision
        if acoustic["fake_probability"] >= 0.95:
            decision = "SCAM"
        else:
            decision = "SCAM" if risk_score > 70 else "SAFE"

        warnings = []
        if TRANSCRIBE_SOURCE != "tool-ab":
            warnings.append("ASR fallback used (install librosa/tool-ab deps for real transcript).")
        if ANALYZE_SOURCE != "tool-ab":
            warnings.append("Acoustic fallback used (install librosa/tool-ab deps for real spoof score).")

        # Step 5: explanation assembly
        explanation = f"""
Transcript: {transcript}

Text flags: {text['flags']}
Acoustic anomaly: {acoustic['anomaly']}

Reasoning:
- Text scam score: {text['scam_score']:.2f}
- Acoustic fake probability: {acoustic['fake_probability']:.2f}
- Fusion weights: acoustic={w_acoustic:.2f}, text={w_text:.2f}
- {explanation_note}

Final Risk Score: {risk_score:.2f}
"""

        llm_explanation = None
        if self._llm_agent is not None:
            if not self._llm_agent.enabled:
                warnings.append("LLM explainability disabled (missing API key).")
            else:
                llm_explanation = self._llm_agent.explain(
                    transcript=transcript,
                    text_flags=text["flags"],
                    text_score=float(text["scam_score"]),
                    acoustic_prob=float(acoustic["fake_probability"]),
                    acoustic_anomaly=str(acoustic["anomaly"]),
                    decision=decision,
                    risk_score=float(risk_score),
                )
                if not llm_explanation:
                    detail = self._llm_agent.last_error
                    if detail:
                        warnings.append(f"LLM explainability failed: {detail}")
                    else:
                        warnings.append("LLM explainability failed (check provider/key/connectivity).")

        if not transcript.strip():
            warnings.append("ASR produced an empty transcript (try WAV, check ffmpeg, or set AGENTIC_ASR_LANGUAGE).")

        warning_block = ""
        if warnings:
            warning_block = "Warnings:\n- " + "\n- ".join(warnings) + "\n\n"

        explanation = warning_block + explanation.strip()

        if llm_explanation:
            explanation = explanation + "\n\nLLM Explainability:\n" + llm_explanation

        return {
            "decision": decision,
            "risk_score": round(risk_score, 2),
            "explanation": explanation.strip(),
            "llm_explanation": llm_explanation,
            "warnings": warnings,
            "tool_sources": {
                "asr": TRANSCRIBE_SOURCE,
                "acoustic": ANALYZE_SOURCE,
            },
            "fusion_details": {
                "asr_present": asr_present,
                "w_acoustic": round(w_acoustic, 2),
                "w_text": round(w_text, 2),
            },
        }


AgenticAIScamDetectionAgent = ScamDetectionAgent