from tools.asr import transcribe
from tools.acoustic import analyze_audio
from tools.text import analyze_text


class ScamDetectionAgent:

    def evaluate(self, audio_path):
        # Step 1: collect info
        transcript = transcribe(audio_path)
        acoustic = analyze_audio(audio_path)
        text = analyze_text(transcript)

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