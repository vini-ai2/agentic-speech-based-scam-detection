# =========================================================
# realtime_tool_c.py
# REALTIME TOOL C INFERENCE
# =========================================================

import math
import torch
import pandas as pd

from flashtext import KeywordProcessor

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# =========================================================
# CONFIG
# =========================================================

MODEL_OUTPUT_DIR = "./models/tool_c_final"

LEXICON_PATH = "./data/scam_lexicon.csv"

MAX_LEN = 128

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"\n[+] Using device: {DEVICE}")

# =========================================================
# LOAD MODEL
# =========================================================

print("\n[+] Loading Tool C model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_OUTPUT_DIR
)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(MODEL_OUTPUT_DIR)
    .to(DEVICE)
)

model.eval()

print("[+] Model loaded.")

# =========================================================
# LOAD LEXICON
# =========================================================

print("\n[+] Loading lexicon...")

lexicon_df = pd.read_csv(
    LEXICON_PATH
)

keyword_processor = KeywordProcessor(
    case_sensitive=False
)

phrase_to_data = {}

for _, row in lexicon_df.iterrows():

    phrase = str(row["phrase"])

    keyword_processor.add_keyword(
        phrase
    )

    phrase_to_data[phrase] = {

        "weight": float(row["weight"]),

        "category": str(row["category"])
    }

print("[+] Lexicon loaded.")

# =========================================================
# FEATURE ENGINE
# =========================================================

def analyze_features(text):

    text_lower = text.lower()

    matched_phrases = (
        keyword_processor.extract_keywords(
            text_lower
        )
    )

    feature_score = 0

    category_scores = {}

    for phrase in matched_phrases:

        data = phrase_to_data.get(phrase)

        if data is None:
            continue

        weight = data["weight"]

        category = data["category"]

        feature_score += weight

        if category not in category_scores:

            category_scores[category] = 0

        category_scores[category] += weight

    normalized = (
        math.log1p(feature_score)
        / math.log1p(10)
    )

    normalized = min(
        normalized,
        1.0
    )

    return {

        "feature_score": round(
            normalized,
            4
        ),

        "matched_phrases":
            matched_phrases[:10],

        "category_scores":
            category_scores,
    }

# =========================================================
# REALTIME PREDICTION
# =========================================================

def predict(text):

    # ================================================
    # FEATURE ANALYSIS
    # ================================================

    feature_data = analyze_features(
        text
    )

    # ================================================
    # BUILD ENHANCED TEXT
    # ================================================

    features = []

    for cat in feature_data[
        "category_scores"
    ]:

        features.append(
            f"{cat}_detected"
        )

    enhanced_text = (

        text

        + " [FEATURES] "

        + " ".join(features)

        + " [MATCHED] "

        + " ".join(
            feature_data[
                "matched_phrases"
            ]
        )
    )

    # ================================================
    # TOKENIZATION
    # ================================================

    inputs = tokenizer(

        enhanced_text,

        return_tensors="pt",

        truncation=True,

        padding=True,

        max_length=MAX_LEN
    )

    inputs = {

        k: v.to(DEVICE)

        for k, v in inputs.items()
    }

    # ================================================
    # MODEL INFERENCE
    # ================================================

    with torch.no_grad():

        outputs = model(
            **inputs
        )

        probs = torch.softmax(

            outputs.logits,

            dim=-1
        )

        semantic_score = (
            probs[0][1].item()
        )

    # ================================================
    # FEATURE SCORE
    # ================================================

    feature_score = (
        feature_data[
            "feature_score"
        ]
    )

    # ================================================
    # ADAPTIVE FUSION
    # ================================================

    if feature_score >= 0.75:

        semantic_weight = 0.70
        feature_weight = 0.30

    elif feature_score >= 0.40:

        semantic_weight = 0.80
        feature_weight = 0.20

    else:

        semantic_weight = 0.90
        feature_weight = 0.10

    final_score = (

        semantic_weight
        * semantic_score

        +

        feature_weight
        * feature_score
    )

    # ================================================
    # RISK LEVEL
    # ================================================

    if final_score >= 0.85:

        risk = "HIGH"

    elif final_score >= 0.60:

        risk = "MEDIUM"

    else:

        risk = "LOW"

    # ================================================
    # OUTPUT
    # ================================================

    print("\n==============================")

    print("\nINPUT:")
    print(text)

    print("\nRESULT:")

    print(
        f"Semantic Score : "
        f"{semantic_score:.4f}"
    )

    print(
        f"Feature Score  : "
        f"{feature_score:.4f}"
    )

    print(
        f"Final Score    : "
        f"{final_score:.4f}"
    )

    print(
        f"Risk Level     : "
        f"{risk}"
    )

    print(
        f"Matched Phrases: "
        f"{feature_data['matched_phrases']}"
    )

    print(
        f"Category Scores: "
        f"{feature_data['category_scores']}"
    )

# =========================================================
# REALTIME LOOP
# =========================================================

print("\n[+] REALTIME TOOL C READY")

print("\nType 'exit' to quit.\n")

while True:

    user_input = input(
        "\nEnter text: "
    )

    if user_input.lower() == "exit":

        break

    predict(user_input)