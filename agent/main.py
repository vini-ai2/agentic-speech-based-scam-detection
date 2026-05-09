import argparse
from pathlib import Path

try:
	from agent import AgenticAIScamDetectionAgent
except Exception:
	from agent import ScamDetectionAgent as AgenticAIScamDetectionAgent


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run the agentic AI scam detection demo")
	parser.add_argument("audio", help="Path to an audio file (wav/mp3/flac)")
	parser.add_argument("--no-llm", action="store_true", help="Disable LLM explainability")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	audio_path = Path(args.audio)
	if not audio_path.exists():
		raise SystemExit(f"Audio file not found: {audio_path}")

	agent = AgenticAIScamDetectionAgent(use_llm=not args.no_llm)
	result = agent.evaluate(str(audio_path))

	print("\n===== SCAM DETECTION RESULT =====\n")

	print(f"Decision     : {result['decision']}")
	print(f"Risk Score   : {result['risk_score']}/100")

	print("\nExplanation:")
	print(result["explanation"])

	print("\n================================\n")


if __name__ == "__main__":
	main()