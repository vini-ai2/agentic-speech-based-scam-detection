# =========================================================
# build_c_final.py
# STABLE PYTORCH + CUDA TOOL C PIPELINE
# =========================================================

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from flashtext import KeywordProcessor

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from sklearn.utils.class_weight import (
    compute_class_weight
)

# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "microsoft/MiniLM-L12-H384-uncased"

FEATURE_DATASET_PATH = "./data/feature_dataset.csv"

LEXICON_PATH = "./data/scam_lexicon.csv"

MODEL_OUTPUT_DIR = "./models/tool_c_final"

MAX_LEN = 128

BATCH_SIZE = 32

EPOCHS = 3

LEARNING_RATE = 2e-5

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"\n[+] Using device: {DEVICE}")

# =========================================================
# LOAD DATASET
# =========================================================

print("\n[+] Loading dataset...")

df = pd.read_csv(
    FEATURE_DATASET_PATH
)

# =========================================================
# CLEAN DATA
# =========================================================

df = df.dropna(
    subset=["text", "label"]
)

df = df[
    df["label"].isin([0, 1])
]

df["text"] = (
    df["text"]
    .astype(str)
)

print("\nDataset Shape:")
print(df.shape)

# =========================================================
# LOAD LEXICON
# =========================================================

print("\n[+] Loading scam lexicon...")

lexicon_df = pd.read_csv(
    LEXICON_PATH
)
# =========================================================
# MANUAL HIGH-RISK PHRASES
# =========================================================

manual_phrases = [

    # Financial
    ("bitcoin", "financial", 2.5),
    ("gift cards", "financial", 2.5),
    ("wire transfer", "financial", 2.5),
    ("bank account", "financial", 2.0),
    ("payment required", "financial", 2.0),

    # Authority
    ("irs", "authority", 3.0),
    ("government", "authority", 2.0),
    ("police", "authority", 2.0),
    ("legal action", "threat", 2.5),
    ("lawsuit", "threat", 2.5),

    # Tech scams
    ("malware", "technical", 2.5),
    ("security alert", "technical", 2.5),
    ("infected", "technical", 2.0),
    ("remote access", "technical", 3.0),

    # Threat
    ("account suspension", "threat", 2.5),
    ("account termination", "threat", 2.5),
    ("arrest warrant", "threat", 3.0),

    # Urgency
    ("immediately", "urgency", 1.5),
    ("urgent", "urgency", 1.5),
    ("act now", "urgency", 2.0),
]

manual_df = pd.DataFrame(

    manual_phrases,

    columns=[
        "phrase",
        "category",
        "weight"
    ]
)

manual_df["spam_ratio"] = 1.0
manual_df["spam_count"] = 999

lexicon_df = pd.concat(

    [lexicon_df, manual_df],

    ignore_index=True
)

# =========================================================
# IMPORTANT UNIGRAMS
# =========================================================

IMPORTANT_UNIGRAMS = {

    "bitcoin",
    "irs",
    "bank",
    "crypto",
    "gift",
    "warrant",
    "arrest",
    "lawsuit",
    "wire",
    "refund",
    "medicare",
    "paypal",
    "amazon",
    "government",
    "police",
}

# =========================================================
# GENERIC WORDS
# =========================================================

GENERIC_WORDS = {

    "required",
    "action",
    "avoid",
    "reply",
    "today",
    "okay",
    "message",
    "txt",
    "mobile",
    "customer",
    "service",
    "claim",
    "receive",
    "hello",
    "thanks",
    "please",
}

# =========================================================
# CLEAN LEXICON
# =========================================================

lexicon_df = lexicon_df[
    lexicon_df["spam_ratio"] >= 0.80
]

lexicon_df = lexicon_df[
    lexicon_df["spam_count"] >= 5
]

lexicon_df = lexicon_df[
    ~lexicon_df["phrase"].isin(
        GENERIC_WORDS
    )
]

lexicon_df["ngram_size"] = (

    lexicon_df["phrase"]
    .str.split()
    .apply(len)
)

lexicon_df = lexicon_df[

    (lexicon_df["ngram_size"] >= 2)

    |

    (
        lexicon_df["phrase"]
        .isin(IMPORTANT_UNIGRAMS)
    )
]

print("\nFiltered Lexicon Size:")
print(len(lexicon_df))

# =========================================================
# FLASH TEXT ENGINE
# =========================================================

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

    # =====================================================
    # LOG NORMALIZATION
    # =====================================================

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
# BUILD ENHANCED TEXT
# =========================================================

print("\n[+] Building semantic inputs...")

def build_enhanced_text(text):

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

    enhanced = (

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

    return enhanced

df["enhanced_text"] = (
    df["text"]
    .apply(build_enhanced_text)
)

# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================

train_df, val_df = train_test_split(

    df,

    test_size=0.2,

    random_state=42,

    stratify=df["label"]
)

print("\nTrain Shape:")
print(train_df.shape)

print("\nValidation Shape:")
print(val_df.shape)

# =========================================================
# CLASS WEIGHTS
# =========================================================

class_weights = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(
        train_df["label"]
    ),

    y=train_df["label"]
)

class_weights = torch.tensor(

    class_weights,

    dtype=torch.float
).to(DEVICE)

print("\nClass Weights:")
print(class_weights)

# =========================================================
# TOKENIZER
# =========================================================

print("\n[+] Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("\n[+] Tokenizer loaded.")

# =========================================================
# PYTORCH DATASET
# =========================================================

class ScamDataset(Dataset):

    def __init__(

        self,

        texts,

        labels,

        tokenizer,

        max_len
    ):

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, idx):

        text = str(
            self.texts[idx]
        )

        label = int(
            self.labels[idx]
        )

        encoding = self.tokenizer(

            text,

            truncation=True,

            padding="max_length",

            max_length=self.max_len,

            return_tensors="pt"
        )

        return {

            "input_ids":
                encoding["input_ids"]
                .squeeze(0),

            "attention_mask":
                encoding["attention_mask"]
                .squeeze(0),

            "labels":
                torch.tensor(
                    label,
                    dtype=torch.long
                )
        }

# =========================================================
# BUILD DATASETS
# =========================================================

print("\n[+] Building PyTorch datasets...")

train_dataset = ScamDataset(

    train_df["enhanced_text"].tolist(),

    train_df["label"].tolist(),

    tokenizer,

    MAX_LEN
)

val_dataset = ScamDataset(

    val_df["enhanced_text"].tolist(),

    val_df["label"].tolist(),

    tokenizer,

    MAX_LEN
)

print("\n[+] Dataset creation complete.")

# =========================================================
# LOAD MODEL
# =========================================================

print("\n[+] Loading MiniLM model...")

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_NAME,
        num_labels=2
    )
)

model.to(DEVICE)

print("\n[+] Model loaded successfully.")

# =========================================================
# METRICS
# =========================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = logits.argmax(
        axis=-1
    )

    precision, recall, f1, _ = (

        precision_recall_fscore_support(

            labels,

            predictions,

            average="binary"
        )
    )

    acc = accuracy_score(
        labels,
        predictions
    )

    return {

        "accuracy": acc,

        "f1": f1,

        "precision": precision,

        "recall": recall,
    }

# =========================================================
# CUSTOM TRAINER
# =========================================================

class WeightedTrainer(Trainer):

    def compute_loss(

        self,

        model,

        inputs,

        return_outputs=False,

        **kwargs
    ):

        labels = inputs.get("labels")

        outputs = model(

            input_ids=inputs[
                "input_ids"
            ],

            attention_mask=inputs[
                "attention_mask"
            ]
        )

        logits = outputs.get(
            "logits"
        )

        loss_fct = nn.CrossEntropyLoss(
            weight=class_weights
        )

        loss = loss_fct(

            logits.view(-1, 2),

            labels.view(-1)
        )

        return (

            (loss, outputs)

            if return_outputs

            else loss
        )

# =========================================================
# TRAINING ARGS
# =========================================================

training_args = TrainingArguments(

    output_dir=MODEL_OUTPUT_DIR,

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=
        BATCH_SIZE,

    per_device_eval_batch_size=
        BATCH_SIZE,

    num_train_epochs=EPOCHS,

    weight_decay=0.01,

    logging_steps=25,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    fp16=torch.cuda.is_available(),
)

# =========================================================
# TRAINER
# =========================================================

trainer = WeightedTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    compute_metrics=compute_metrics,
)

# =========================================================
# TRAIN
# =========================================================

print("\n[+] Starting training...")

trainer.train()

# =========================================================
# EVALUATE
# =========================================================

print("\n[+] Evaluating model...")

metrics = trainer.evaluate()

print("\nFINAL METRICS:")
print(metrics)

# =========================================================
# SAVE MODEL
# =========================================================

print("\n[+] Saving model...")

trainer.save_model(
    MODEL_OUTPUT_DIR
)

tokenizer.save_pretrained(
    MODEL_OUTPUT_DIR
)

print("\n[+] Model Saved:")
print(MODEL_OUTPUT_DIR)

# =========================================================
# REALTIME INFERENCE ENGINE
# =========================================================

print("\n[+] Building inference engine...")

class ToolCScamDetector:

    def __init__(self):

        self.device = DEVICE

        self.tokenizer = (

            AutoTokenizer

            .from_pretrained(
                MODEL_OUTPUT_DIR
            )
        )

        self.model = (

            AutoModelForSequenceClassification

            .from_pretrained(
                MODEL_OUTPUT_DIR
            )

            .to(self.device)
        )

        self.model.eval()

    # =====================================================
    # PREDICT
    # =====================================================

    def predict(self, text):

        # ================================================
        # FEATURE ANALYSIS
        # ================================================

        feature_data = analyze_features(
            text
        )

        # ================================================
        # ENHANCED TEXT
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
        # TOKENIZE
        # ================================================

        inputs = self.tokenizer(

            enhanced_text,

            return_tensors="pt",

            truncation=True,

            padding=True,

            max_length=MAX_LEN
        )

        inputs = {

            k: v.to(self.device)

            for k, v in inputs.items()
        }

        # ================================================
        # MODEL INFERENCE
        # ================================================

        with torch.no_grad():

            outputs = self.model(
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
        # ADAPTIVE CONFIDENCE FUSION
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
                ],

            "category_scores":

                feature_data[
                    "category_scores"
                ]
        }

# =========================================================
# QUICK TEST
# =========================================================

# =========================================================
# INTEGRATED TEST SUITE
# =========================================================

print("\n[+] Running integrated Tool C test suite...\n")

test_cases = [

    {
        "name": "Prize Scam",

        "text": """
        Congratulations! You have won a 2000 prize guaranteed.
        Reply YES to claim your cash award now.
        """
    },

    {
        "name": "Urgency Scam",

        "text": """
        URGENT! We are trying to contact you regarding
        your mobile number awarded 5000 cash.
        Call now to claim.
        """
    },

    {
        "name": "Banking Scam",

        "text": """
        Your bank account statement shows unusual activity.
        Immediate action required to avoid suspension.
        """
    },

    {
        "name": "Lottery Scam",

        "text": """
        You are the lucky winner of a guaranteed cash prize.
        Reply within 24 hrs to receive your reward.
        """
    },

    {
        "name": "Recharge Scam",

        "text": """
        Recharge now and receive free talktime bonus.
        Limited offer valid for 24 hrs only.
        """
    },

    {
        "name": "IRS Bitcoin Scam",

        "text": """
        This is the IRS.
        Immediate Bitcoin payment required
        to avoid legal action.
        """
    },

    {
        "name": "Tech Support Scam",

        "text": """
        Microsoft security alert.
        Your device is infected with malware.
        Call support immediately.
        """
    },

    {
        "name": "Gift Card Scam",

        "text": """
        Purchase gift cards immediately and send
        the codes to avoid account termination.
        """
    },

    {
        "name": "Normal Friendly Message",

        "text": """
        Hey, are we still meeting tomorrow
        for lunch at 1 PM?
        """
    },

    {
        "name": "Academic Message",

        "text": """
        Can you send me the notes from today's class?
        I'll review them tonight.
        """
    }
]

# =========================================================
# RUN TESTS
# =========================================================

detector = ToolCScamDetector()

for i, test in enumerate(test_cases):

    print("=" * 60)

    print(f"\nTEST {i+1}: {test['name']}")

    print("\nINPUT:")
    print(test["text"])

    result = detector.predict(
        test["text"]
    )

    print("\nOUTPUT:")
    print(result)

    print("\n")

# =========================================================
# TOOL C VISUALIZATION ENGINE
# =========================================================

print("\n[+] Generating Tool C visualizations...")

# =========================================================
# COLLECT RESULTS
# =========================================================

test_names = []

semantic_scores = []

feature_scores = []

final_scores = []

risk_levels = []

for test in test_cases:

    result = detector.predict(
        test["text"]
    )

    test_names.append(
        test["name"]
    )

    semantic_scores.append(
        result["semantic_score"]
    )

    feature_scores.append(
        result["feature_score"]
    )

    final_scores.append(
        result["final_score"]
    )

    risk_levels.append(
        result["risk_level"]
    )

# =========================================================
# CREATE OUTPUT FOLDER
# =========================================================

os.makedirs(
    "./tool_c_graphs",
    exist_ok=True
)

# =========================================================
# GRAPH 1
# SEMANTIC VS FEATURE VS FINAL
# =========================================================

x = np.arange(len(test_names))

width = 0.25

plt.figure(figsize=(14, 7))

plt.bar(
    x - width,
    semantic_scores,
    width,
    label="Semantic"
)

plt.bar(
    x,
    feature_scores,
    width,
    label="Feature"
)

plt.bar(
    x + width,
    final_scores,
    width,
    label="Final"
)

plt.xticks(
    x,
    test_names,
    rotation=25
)

plt.ylim(0, 1.05)

plt.ylabel("Score")

plt.title(
    "Tool C: Semantic vs Feature vs Final Risk"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "./tool_c_graphs/tool_c_score_breakdown.png"
)

plt.close()

# =========================================================
# GRAPH 2
# FINAL RISK SCORES
# =========================================================

plt.figure(figsize=(12, 6))

plt.bar(
    test_names,
    final_scores
)

plt.axhline(
    y=0.60,
    linestyle="--",
    label="Medium Risk"
)

plt.axhline(
    y=0.85,
    linestyle="--",
    label="High Risk"
)

plt.ylabel("Final Risk Score")

plt.title(
    "Tool C Final Scam Risk Scores"
)

plt.xticks(rotation=25)

plt.legend()

plt.tight_layout()

plt.savefig(
    "./tool_c_graphs/tool_c_final_risk.png"
)

plt.close()

# =========================================================
# GRAPH 3
# MODEL PERFORMANCE METRICS
# =========================================================

metric_names = [

    "Accuracy",
    "Precision",
    "Recall",
    "F1"
]

metric_values = [

    metrics["eval_accuracy"],

    metrics["eval_precision"],

    metrics["eval_recall"],

    metrics["eval_f1"]
]

plt.figure(figsize=(8, 6))

plt.bar(
    metric_names,
    metric_values
)

plt.ylim(0.8, 1.0)

plt.ylabel("Score")

plt.title(
    "Tool C Validation Metrics"
)

for i, v in enumerate(metric_values):

    plt.text(
        i,
        v + 0.003,
        f"{v:.3f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "./tool_c_graphs/tool_c_metrics.png"
)

plt.close()

# =========================================================
# GRAPH 4
# RISK LEVEL DISTRIBUTION
# =========================================================

risk_counts = {

    "HIGH": risk_levels.count("HIGH"),

    "MEDIUM": risk_levels.count("MEDIUM"),

    "LOW": risk_levels.count("LOW"),
}

plt.figure(figsize=(7, 7))

plt.pie(

    risk_counts.values(),

    labels=risk_counts.keys(),

    autopct="%1.1f%%"
)

plt.title(
    "Tool C Risk Level Distribution"
)

plt.savefig(
    "./tool_c_graphs/tool_c_risk_distribution.png"
)

plt.close()

# =========================================================
# DONE
# =========================================================

print("\n[+] Tool C graphs saved to:")

print("./tool_c_graphs/")

print("\n[+] TOOL C READY")