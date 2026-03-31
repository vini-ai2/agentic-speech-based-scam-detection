def analyze_text(text):
    keywords = ["transfer", "urgent", "bank", "otp", "immediately"]

    flags = [word for word in keywords if word in text.lower()]
    score = len(flags) / len(keywords)

    return {
        "scam_score": score,
        "flags": flags
    }