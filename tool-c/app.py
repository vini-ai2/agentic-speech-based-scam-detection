# =========================================================
# app.py
# TOOL C MINIMAL UI
# =========================================================

import math
import torch
import pandas as pd
import streamlit as st

from flashtext import KeywordProcessor

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(

    page_title="Tool C Scam Detector",

    page_icon="🛡️",

    layout="centered"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""

<style>

/* =====================================================
BACKGROUND
===================================================== */

.stApp {

    background: linear-gradient(
        135deg,
        #050816 0%,
        #0a1026 50%,
        #050816 100%
    );

    color: white;
}

/* =====================================================
TITLE
===================================================== */

.main-title {

    font-size: 42px;

    font-weight: 700;

    color: #2dd4ff;

    margin-bottom: 5px;
}

.subtitle {

    color: #9ca3af;

    margin-bottom: 30px;
}

/* =====================================================
CARD
===================================================== */

.card {

    background: rgba(10, 15, 35, 0.95);

    border-radius: 18px;

    padding: 30px;

    border: 1px solid rgba(45, 212, 255, 0.12);

    box-shadow: 0 0 30px rgba(0,0,0,0.35);
}

/* =====================================================
TEXT AREA
===================================================== */

textarea {

    background-color: #0f172a !important;

    color: white !important;

    border-radius: 12px !important;

    border: 1px solid rgba(45,212,255,0.2) !important;
}

/* =====================================================
BUTTON
===================================================== */

.stButton > button {

    background: linear-gradient(
        90deg,
        #22d3ee,
        #2563eb
    );

    color: white;

    border: none;

    border-radius: 12px;

    font-size: 18px;

    font-weight: 600;

    padding: 12px 25px;

    width: 100%;
}

/* =====================================================
RESULT BOX
===================================================== */

.result-box {

    background: #0f172a;

    padding: 20px;

    border-radius: 15px;

    margin-top: 20px;

    border: 1px solid rgba(45,212,255,0.15);
}

/* =====================================================
RISK COLORS
===================================================== */

.high-risk {

    color: #ef4444;

    font-weight: 700;
}

.medium-risk {

    color: #f59e0b;

    font-weight: 700;
}

.low-risk {

    color: #22c55e;

    font-weight: 700;
}

</style>

""", unsafe_allow_html=True)

# =========================================================
# CONFIG
# =========================================================

MODEL_DIR = "./models/tool_c_final"

LEXICON_PATH = "./data/scam_lexicon.csv"

MAX_LEN = 128

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    tokenizer = (
        AutoTokenizer
        .from_pretrained(MODEL_DIR)
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(MODEL_DIR)
        .to(DEVICE)
    )

    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

# =========================================================
# LOAD LEXICON
# =========================================================

@st.cache_resource
def load_lexicon():

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

    return keyword_processor, phrase_to_data

keyword_processor, phrase_to_data = (
    load_lexicon()
)

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

        data = phrase_to_data.get(
            phrase
        )

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

        "feature_score":
            round(normalized, 4),

        "matched_phrases":
            matched_phrases[:10],

        "category_scores":
            category_scores,
    }

# =========================================================
# PREDICT
# =========================================================

def predict(text):

    feature_data = analyze_features(
        text
    )

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

    feature_score = (
        feature_data[
            "feature_score"
        ]
    )

    # =====================================================
    # ADAPTIVE FUSION
    # =====================================================

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

    # =====================================================
    # RISK LEVEL
    # =====================================================

    if final_score >= 0.85:

        risk = "HIGH"

    elif final_score >= 0.60:

        risk = "MEDIUM"

    else:

        risk = "LOW"

    return {

        "semantic_score":
            round(
                semantic_score,
                4
            ),

        "feature_score":
            round(
                feature_score,
                4
            ),

        "final_score":
            round(
                final_score,
                4
            ),

        "risk_level":
            risk,

        "matched_phrases":

            feature_data[
                "matched_phrases"
            ]
    }

# =========================================================
# UI
# =========================================================

st.markdown(

    '<div class="main-title">'
    'Agentic Scam Detector'
    '</div>',

    unsafe_allow_html=True
)

st.markdown(

    '<div class="subtitle">'
    'Tool C — Semantic + Feature Fusion'
    '</div>',

    unsafe_allow_html=True
)

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

user_input = st.text_area(

    "Enter suspicious message or transcript:",

    height=180,

    placeholder="""
Example:
This is the IRS.
Immediate Bitcoin payment required
to avoid legal action.
"""
)

analyze = st.button(
    "Analyze Message"
)

if analyze and user_input.strip():

    result = predict(user_input)

    risk_class = {

        "HIGH": "high-risk",

        "MEDIUM": "medium-risk",

        "LOW": "low-risk"
    }[
        result["risk_level"]
    ]

    st.markdown(

        f"""
        <div class="result-box">

        <h3>Analysis Result</h3>

        <p><b>Semantic Score:</b>
        {result['semantic_score']}</p>

        <p><b>Feature Score:</b>
        {result['feature_score']}</p>

        <p><b>Final Risk Score:</b>
        {result['final_score']}</p>

        <p><b>Risk Level:</b>
        <span class="{risk_class}">
        {result['risk_level']}
        </span></p>

        <p><b>Matched Phrases:</b>
        {result['matched_phrases']}</p>

        </div>
        """,

        unsafe_allow_html=True
    )

st.markdown(
    "</div>",
    unsafe_allow_html=True
)