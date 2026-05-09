import re


_WORD_RE = re.compile(r"[a-z0-9']+")
_CARD_NUMBER_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_OTP_RE = re.compile(r"\b(?:otp|one[- ]?time code|verification code|security code)\b.*?\b\d{4,8}\b|\b\d{4,8}\b.*?\b(?:otp|one[- ]?time code|verification code|security code)\b")
_CVV_RE = re.compile(r"\b(?:cvv|cvc|card verification value|security code)\b")
_EXPIRY_RE = re.compile(r"\b(?:expiry|expiration|exp date|valid thru|valid through)\b")
_PRESSURE_RE = re.compile(r"\b(?:urgent|immediately|right now|today|asap|suspend|suspended|account closed|account suspension)\b")


_CONCEPTS = {
    "card": {"card", "visa", "master", "mastercard", "credit", "debit"},
    "verification": {"verify", "confirm", "authorize", "authorized", "holder", "identity", "identity"},
    "security": {"security", "code", "cvv", "cvc", "otp", "pin", "password", "passcode"},
    "pressure": {"urgent", "immediately", "asap", "now", "today", "suspend", "suspended", "suspension"},
    "money": {"bank", "transfer", "account", "payment", "wire", "transaction"},
}


_PHRASES = [
    "security code",
    "confirmation code",
    "verification code",
    "authorized card holder",
    "confirm your card",
    "confirm your card with me",
    "verify your identity",
    "verify your card",
    "card number",
    "account suspension",
    "account closed",
    "your visa or your master card",
    "your visa or your mastercard",
]


def _normalize_tokens(text):
    return set(_WORD_RE.findall(text.lower()))


def _semantic_similarity(tokens):
    concept_hits = {}
    for concept, synonyms in _CONCEPTS.items():
        overlap = sorted(synonyms.intersection(tokens))
        if overlap:
            concept_hits[concept] = overlap

    # Coverage across distinct scam concepts is more important than raw keyword count.
    coverage = len(concept_hits) / float(len(_CONCEPTS))
    pressure_bonus = 0.15 if "pressure" in concept_hits else 0.0
    verification_bonus = 0.12 if "verification" in concept_hits else 0.0
    security_bonus = 0.12 if "security" in concept_hits else 0.0
    money_bonus = 0.10 if "money" in concept_hits else 0.0
    card_bonus = 0.10 if "card" in concept_hits else 0.0

    return min(1.0, coverage + pressure_bonus + verification_bonus + security_bonus + money_bonus + card_bonus), concept_hits


def analyze_text(text):
    t = text.lower()
    tokens = _normalize_tokens(t)

    matched_phrases = [phrase for phrase in _PHRASES if phrase in t]

    regex_hits = []
    if _CARD_NUMBER_RE.search(t):
        regex_hits.append("possible_card_number")
    if _OTP_RE.search(t):
        regex_hits.append("possible_otp_or_verification_code")
    if _CVV_RE.search(t):
        regex_hits.append("possible_cvv_or_security_code")
    if _EXPIRY_RE.search(t):
        regex_hits.append("possible_expiry_or_validity_request")
    if _PRESSURE_RE.search(t):
        regex_hits.append("pressure_or_urgency")

    semantic_score, concept_hits = _semantic_similarity(tokens)

    # Weight the signals so that explicit patterns dominate, then semantic coverage,
    # then phrase matches. Cap at 1.0.
    pattern_score = min(1.0, 0.28 * len(regex_hits))
    phrase_score = min(1.0, 0.12 * len(matched_phrases))
    score = min(1.0, pattern_score + phrase_score + 0.45 * semantic_score)

    flags = matched_phrases + regex_hits + [f"concept:{name}={','.join(values)}" for name, values in concept_hits.items()]

    return {
        "scam_score": score,
        "flags": flags,
        "matched_phrases": matched_phrases,
        "regex_hits": regex_hits,
        "semantic_score": semantic_score,
        "semantic_concepts": concept_hits,
    }