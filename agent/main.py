from agent import ScamDetectionAgent

agent = ScamDetectionAgent()

result = agent.evaluate("scam.mp3")

print("\n===== SCAM DETECTION RESULT =====\n")

print(f"Decision     : {result['decision']}")
print(f"Risk Score   : {result['risk_score']}/100")

print("\nExplanation:")
print(result["explanation"])

print("\n================================\n")