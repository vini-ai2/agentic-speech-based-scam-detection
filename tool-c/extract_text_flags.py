import os
import glob
import re
from collections import Counter, defaultdict

import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

# =========================================================
# OUTPUT FILES
# =========================================================

OUTPUT_LEXICON = "./data/scam_lexicon.csv"
OUTPUT_FEATURE_DATASET = "./data/feature_dataset.csv"

os.makedirs("./data", exist_ok=True)

# =========================================================
# TEXT CLEANER
# =========================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# =========================================================
# LABEL NORMALIZER
# =========================================================

def normalize_label(label):

    label = str(label).lower()

    if label in ["spam", "scam", "fraud", "1"]:
        return 1

    if label in ["ham", "safe", "legit", "0"]:
        return 0

    return -1

# =========================================================
# STORAGE
# =========================================================

texts = []
labels = []

# =========================================================
# LOAD 1 — SMSSpamCollection
# =========================================================

print("\n[+] Loading SMSSpamCollection...")

sms_path = "../datasets/sms+spam+collection/SMSSpamCollection"

df_sms = pd.read_csv(
    sms_path,
    sep="\t",
    header=None,
    names=["label", "text"],
    encoding="latin-1"
)

for _, row in tqdm(df_sms.iterrows(), total=len(df_sms)):

    texts.append(clean_text(row["text"]))
    labels.append(normalize_label(row["label"]))

# =========================================================
# LOAD 2 — spam.csv
# =========================================================

print("\n[+] Loading spam.csv...")

spam_csv_path = "../datasets/archive/spam.csv"

try:

    df_spam = pd.read_csv(
        spam_csv_path,
        encoding="latin-1"
    )

    print("Columns:", df_spam.columns)

    # spam.csv specifically uses:
    # v1 = label
    # v2 = text

    text_col = "v2"
    label_col = "v1"

    for _, row in tqdm(df_spam.iterrows(), total=len(df_spam)):

        texts.append(clean_text(row[text_col]))
        labels.append(normalize_label(row[label_col]))

except Exception as e:

    print(f"[!] Failed spam.csv: {e}")

# =========================================================
# LOAD 3 — IIIT-D DATASET
# =========================================================

print("\n[+] Loading IIIT-D dataset...")

ham_files = glob.glob(
    "../datasets/IIIT-D_SMS_Dataset/IIIT-D_SMS_Dataset/Ham SMSes/*.txt"
)

spam_files = glob.glob(
    "../datasets/IIIT-D_SMS_Dataset/IIIT-D_SMS_Dataset/Spam SMSes/*.txt"
)

print(f"Ham files: {len(ham_files)}")
print(f"Spam files: {len(spam_files)}")

# ---------------------------
# HAM
# ---------------------------

for file in tqdm(ham_files):

    try:

        with open(file, "r", encoding="latin-1") as f:

            text = clean_text(f.read())

        texts.append(text)
        labels.append(0)

    except Exception as e:

        print(f"Failed: {file} | {e}")

# ---------------------------
# SPAM
# ---------------------------

for file in tqdm(spam_files):

    try:

        with open(file, "r", encoding="latin-1") as f:

            text = clean_text(f.read())

        texts.append(text)
        labels.append(1)

    except Exception as e:

        print(f"Failed: {file} | {e}")

# =========================================================
# CREATE MASTER DATAFRAME
# =========================================================

master_df = pd.DataFrame({
    "text": texts,
    "label": labels
})

master_df.drop_duplicates(subset=["text"], inplace=True)

master_df = master_df[
    master_df["text"].str.len() > 3
]

print("\n[+] MASTER DATASET")
print(master_df.head())

print("\nShape:")
print(master_df.shape)

print("\nLabel Distribution:")
print(master_df["label"].value_counts())

# =========================================================
# BUILD SCAM LEXICON
# =========================================================

print("\n[+] Building scam lexicon...")

# only spam/scam texts
spam_texts = master_df[
    master_df["label"] == 1
]["text"].tolist()

ham_texts = master_df[
    master_df["label"] == 0
]["text"].tolist()

# =========================================================
# EXTRACT NGRAMS
# =========================================================

vectorizer = CountVectorizer(
    ngram_range=(1, 3),
    stop_words="english",
    min_df=3
)

X_spam = vectorizer.fit_transform(spam_texts)

spam_vocab = vectorizer.get_feature_names_out()

spam_counts = X_spam.sum(axis=0).A1

spam_freq = dict(zip(spam_vocab, spam_counts))

# =========================================================
# HAM FREQUENCIES
# =========================================================

ham_vectorizer = CountVectorizer(
    vocabulary=spam_vocab
)

X_ham = ham_vectorizer.fit_transform(ham_texts)

ham_counts = X_ham.sum(axis=0).A1

ham_freq = dict(zip(spam_vocab, ham_counts))

# =========================================================
# BUILD LEXICON TABLE
# =========================================================

lexicon_rows = []

for phrase in spam_vocab:

    spam_count = spam_freq.get(phrase, 0)
    ham_count = ham_freq.get(phrase, 0)

    total = spam_count + ham_count

    if total == 0:
        continue

    spam_ratio = spam_count / total

    # Ignore weak indicators
    if spam_count < 3:
        continue

    # Auto-category detection
    category = "general"

    if any(x in phrase for x in [
        "urgent", "immediately", "now", "today"
    ]):
        category = "urgency"

    elif any(x in phrase for x in [
        "bitcoin", "payment", "gift card",
        "bank", "refund", "credit"
    ]):
        category = "financial"

    elif any(x in phrase for x in [
        "irs", "government", "police",
        "security", "federal"
    ]):
        category = "authority"

    elif any(x in phrase for x in [
        "arrest", "lawsuit", "warrant",
        "legal"
    ]):
        category = "threat"

    elif any(x in phrase for x in [
        "virus", "malware", "ip",
        "windows"
    ]):
        category = "technical"

    weight = round(
        spam_ratio * min(spam_count / 10, 1.0),
        4
    )

    lexicon_rows.append({
        "phrase": phrase,
        "spam_count": int(spam_count),
        "ham_count": int(ham_count),
        "spam_ratio": round(spam_ratio, 4),
        "weight": weight,
        "category": category
    })

# =========================================================
# CREATE LEXICON DATAFRAME
# =========================================================

lexicon_df = pd.DataFrame(lexicon_rows)

lexicon_df.sort_values(
    by=["weight", "spam_count"],
    ascending=False,
    inplace=True
)

# =========================================================
# SAVE LEXICON
# =========================================================

lexicon_df.to_csv(
    OUTPUT_LEXICON,
    index=False
)

# =========================================================
# FEATURE ENGINE USING LEXICON
# =========================================================

print("\n[+] Building feature dataset...")

feature_rows = []

top_lexicon = lexicon_df.head(3000)

for _, row in tqdm(master_df.iterrows(), total=len(master_df)):

    text = row["text"]
    label = row["label"]

    matched_phrases = []
    feature_score = 0

    category_scores = defaultdict(float)

    for _, lex_row in top_lexicon.iterrows():

        phrase = lex_row["phrase"]

        if phrase in text:

            weight = lex_row["weight"]
            category = lex_row["category"]

            feature_score += weight

            category_scores[category] += weight

            matched_phrases.append(phrase)

    feature_rows.append({

        "text": text,

        "label": label,

        "feature_score": round(feature_score, 4),

        "urgency_score": round(
            category_scores["urgency"], 4
        ),

        "financial_score": round(
            category_scores["financial"], 4
        ),

        "authority_score": round(
            category_scores["authority"], 4
        ),

        "threat_score": round(
            category_scores["threat"], 4
        ),

        "technical_score": round(
            category_scores["technical"], 4
        ),

        "matched_phrases": ", ".join(
            matched_phrases[:20]
        )
    })

# =========================================================
# FINAL FEATURE DATASET
# =========================================================

feature_df = pd.DataFrame(feature_rows)

feature_df.to_csv(
    OUTPUT_FEATURE_DATASET,
    index=False
)

# =========================================================
# DONE
# =========================================================

print("\n[+] DONE")

print(f"\nLexicon saved to:")
print(OUTPUT_LEXICON)

print(f"\nFeature dataset saved to:")
print(OUTPUT_FEATURE_DATASET)

print("\nTop scam indicators:")
print(
    lexicon_df.head(20)[
        [
            "phrase",
            "spam_count",
            "spam_ratio",
            "weight",
            "category"
        ]
    ]
)