# =========================================================
# build_c_final.py
# HYBRID REALTIME TOOL C PIPELINE
# =========================================================

import os
import torch
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from datasets import Dataset

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from sklearn.model_selection import train_test_split

from flashtext import KeywordProcessor

# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "microsoft/MiniLM-L12-H384-uncased"

FEATURE_DATASET_PATH = "./data/feature_dataset.csv"

LEXICON_PATH = "./data/scam_lexicon.csv"

MODEL_OUTPUT_DIR = "./models/tool_c_final"

MAX_LEN = 128

BATCH_SIZE = 32

EPOCHS = 2

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

print("\n[+] Loading feature dataset...")

df = pd.read_csv(FEATURE_DATASET_PATH)

print(df.head())

print("\nShape:")
print(df.shape)

# =========================================================
# CLEAN DATA
# =========================================================

df = df.dropna(subset=["text", "label"])

df = df[df["label"].isin([0, 1])]

df["text"] = df["text"].astype(str)

# =========================================================
# BUILD ENHANCED INPUT
# =========================================================

print("\n[+] Creating enhanced semantic inputs...")

def build_enhanced_text(row):

    semantic_text = row["text"]

    features = []

    if row["urgency_score"] > 0:
        features.append("urgency_detected")

    if row["financial_score"] > 0:
        features.append("financial_pressure")

    if row["authority_score"] > 0:
        features.append("authority_impersonation")

    if row["threat_score"] > 0:
        features.append("threat_language")

    if row["technical_score"] > 0:
        features.append("technical_scam")

    matched = str(row.get("matched_phrases", ""))

    augmented = (
        semantic_text
        + " [FEATURES] "
        + " ".join(features)
        + " [MATCHED] "
        + matched
    )

    return augmented

df["enhanced_text"] = df.apply(
    build_enhanced_text,
    axis=1
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

print("\nTrain shape:", train_df.shape)
print("Validation shape:", val_df.shape)

# =========================================================
# TOKENIZER
# =========================================================

print("\n[+] Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

# =========================================================
# TOKENIZATION
# =========================================================

def tokenize(batch):

    return tokenizer(
        batch["enhanced_text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )

# =========================================================
# HUGGINGFACE DATASETS
# =========================================================

train_dataset = Dataset.from_pandas(
    train_df[["enhanced_text", "label"]]
)

val_dataset = Dataset.from_pandas(
    val_df[["enhanced_text", "label"]]
)

train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

val_dataset = val_dataset.map(
    tokenize,
    batched=True
)

# =========================================================
# FORMAT DATASETS
# =========================================================

train_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "label"
    ]
)

val_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "label"
    ]
)

# =========================================================
# LOAD MODEL
# =========================================================

print("\n[+] Loading MiniLM model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.to(DEVICE)

# =========================================================
# METRICS
# =========================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = logits.argmax(axis=-1)

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
# TRAINING ARGUMENTS
# =========================================================

training_args = TrainingArguments(

    output_dir=MODEL_OUTPUT_DIR,

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    num_train_epochs=EPOCHS,

    weight_decay=0.01,

    logging_dir="./logs",

    logging_steps=50,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    fp16=torch.cuda.is_available(),
)

# =========================================================
# TRAINER
# =========================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    compute_metrics=compute_metrics,
)

# =========================================================
# TRAIN MODEL
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

trainer.save_model(MODEL_OUTPUT_DIR)

tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

print(f"\n[+] Model saved to:")
print(MODEL_OUTPUT_DIR)

# =========================================================
# TOOL C INFERENCE ENGINE
# =========================================================

print("\n[+] Building Tool C inference engine...")

class ToolCScamDetector:

    def __init__(self):

        self.device = DEVICE

        # ================================================
        # TOKENIZER
        # ================================================

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_OUTPUT_DIR
        )

        # ================================================
        # MODEL
        # ================================================

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(MODEL_OUTPUT_DIR)
            .to(self.device)
        )

        self.model.eval()

        # ================================================
        # LOAD LEXICON
        # ================================================

        print("\n[+] Loading scam lexicon...")

        self.lexicon_df = pd.read_csv(
            LEXICON_PATH
        )

        # ================================================
        # FLASH TEXT KEYWORD ENGINE
        # ================================================

        self.keyword_processor = KeywordProcessor(
            case_sensitive=False
        )

        self.phrase_to_data = {}

        for _, row in self.lexicon_df.iterrows():

            phrase = str(row["phrase"])

            self.keyword_processor.add_keyword(
                phrase
            )

            self.phrase_to_data[phrase] = {

                "weight": float(row["weight"]),

                "category": str(row["category"])
            }

    # =====================================================
    # FEATURE ANALYSIS
    # =====================================================

    def analyze_features(self, text):

        text_lower = text.lower()

        matched_phrases = (
            self.keyword_processor.extract_keywords(
                text_lower
            )
        )

        feature_score = 0

        category_scores = {}

        for phrase in matched_phrases:

            data = self.phrase_to_data.get(phrase)

            if data is None:
                continue

            weight = data["weight"]

            category = data["category"]

            feature_score += weight

            if category not in category_scores:
                category_scores[category] = 0

            category_scores[category] += weight

        normalized = min(feature_score, 1.0)

        return {

            "feature_score": round(normalized, 4),

            "matched_phrases": matched_phrases[:20],

            "category_scores": category_scores,
        }

    # =====================================================
    # PREDICTION
    # =====================================================

    def predict(self, text):

        # ================================================
        # FEATURE ANALYSIS
        # ================================================

        feature_data = self.analyze_features(text)

        # ================================================
        # BUILD FEATURE CONTEXT
        # ================================================

        features = []

        for cat in feature_data["category_scores"]:

            features.append(f"{cat}_detected")

        enhanced_text = (
            text
            + " [FEATURES] "
            + " ".join(features)
            + " [MATCHED] "
            + " ".join(
                feature_data["matched_phrases"]
            )
        )

        # ================================================
        # TOKENIZATION
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

            outputs = self.model(**inputs)

            probs = torch.softmax(
                outputs.logits,
                dim=-1
            )

            semantic_score = (
                probs[0][1].item()
            )

        # ================================================
        # HYBRID CONFIDENCE FUSION
        # ================================================

        semantic_weight = 0.75
        feature_weight = 0.25

        final_score = (
            semantic_weight * semantic_score
            + feature_weight *
            feature_data["feature_score"]
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
        # FINAL OUTPUT
        # ================================================

        return {

            "semantic_score": round(
                semantic_score,
                4
            ),

            "feature_score": round(
                feature_data["feature_score"],
                4
            ),

            "final_score": round(
                final_score,
                4
            ),

            "risk_level": risk,

            "matched_phrases":
                feature_data["matched_phrases"],

            "category_scores":
                feature_data["category_scores"],
        }

# =========================================================
# QUICK TEST
# =========================================================

print("\n[+] Running quick inference test...")

detector = ToolCScamDetector()

sample = """
This is the IRS. Immediate payment
through Bitcoin is required
to avoid legal action.
"""

result = detector.predict(sample)

print("\nTEST RESULT:")
print(result)

print("\n[+] Tool C FINAL BUILD COMPLETE")