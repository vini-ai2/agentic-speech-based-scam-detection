import os
import glob
import pandas as pd
from tqdm import tqdm

# =========================================================
# FEATURE CONFIG
# =========================================================

SCAM_FEATURES = {

    "urgency": {
        "urgent": 2,
        "immediately": 3,
        "right now": 4,
        "today": 2,
        "last warning": 5,
        "final notice": 5,
        "act now": 4,
        "within 24 hours": 5,
    },

    "authority": {
        "irs": 6,
        "social security": 6,
        "medicare": 5,
        "police": 5,
        "federal": 5,
        "government": 4,
        "department of justice": 7,
    },

    "financial": {
        "wire transfer": 7,
        "bitcoin": 8,
        "gift card": 8,
        "bank account": 5,
        "credit card": 4,
        "payment": 3,
        "refund": 2,
    },

    "threat": {
        "arrest": 8,
        "lawsuit": 6,
        "warrant": 7,
        "legal action": 6,
        "suspended": 5,
        "terminated": 4,
        "prosecution": 7,
    },

    "technical": {
        "virus": 5,
        "malware": 6,
        "remote access": 7,
        "ip address": 5,
        "windows support": 5,
        "security alert": 4,
    },

    "emotional": {
        "help me": 4,
        "emergency": 5,
        "don't tell": 6,
        "please help": 4,
        "family member": 3,
        "hospital": 3,
    }
}

# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(text):

    text_lower = str(text).lower()

    feature_scores = {}
    matched_terms = []

    total_score = 0

    for category, keywords in SCAM_FEATURES.items():

        category_score = 0

        for phrase, weight in keywords.items():

            if phrase in text_lower:
                category_score += weight
                matched_terms.append(phrase)

        feature_scores[f"{category}_score"] = category_score
        total_score += category_score

    feature_scores["total_feature_score"] = total_score
    feature_scores["matched_terms"] = ", ".join(matched_terms)

    return feature_scores

# =========================================================
# LABEL NORMALIZATION
# =========================================================

def normalize_label(label):

    label = str(label).lower()

    if label in ["spam", "scam", "fraud", "1"]:
        return 1

    if label in ["ham", "safe", "legit", "0"]:
        return 0

    return -1

# =========================================================
# MAIN STORAGE
# =========================================================

all_rows = []

# =========================================================
# 1. SMS SPAM COLLECTION
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

    text = str(row["text"])
    label = normalize_label(row["label"])

    features = extract_features(text)

    all_rows.append({
        "text": text,
        "label": label,
        **features
    })
    
# =========================================================
# 2. spam.csv DATASET
# =========================================================

print("\n[+] Loading spam.csv...")

spam_csv_path = "../datasets/spam.csv"

try:

    df_spam_csv = pd.read_csv(
        spam_csv_path,
        encoding="latin-1"
    )

    print(df_spam_csv.columns)

    # Detect likely columns
    text_col = None
    label_col = None

    for col in df_spam_csv.columns:

        col_lower = col.lower()

        if any(x in col_lower for x in [
            "text", "message", "sms", "body"
        ]):
            text_col = col

        if any(x in col_lower for x in [
            "label", "class", "type", "spam"
        ]):
            label_col = col

    print(f"Detected text column: {text_col}")
    print(f"Detected label column: {label_col}")

    for _, row in tqdm(
        df_spam_csv.iterrows(),
        total=len(df_spam_csv)
    ):

        text = str(row[text_col])

        label = normalize_label(row[label_col])

        features = extract_features(text)

        all_rows.append({
            "text": text,
            "label": label,
            **features
        })

except Exception as e:

    print(f"[!] Failed loading spam.csv: {e}")  

# =========================================================
# 3. IIIT-D DATASET
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

# HAM
for file in tqdm(ham_files):

    try:
        with open(file, "r", encoding="latin-1") as f:
            text = f.read().strip()

        features = extract_features(text)

        all_rows.append({
            "text": text,
            "label": 0,
            **features
        })

    except Exception as e:
        print(f"Failed: {file} | {e}")

# SPAM
for file in tqdm(spam_files):

    try:
        with open(file, "r", encoding="latin-1") as f:
            text = f.read().strip()

        features = extract_features(text)

        all_rows.append({
            "text": text,
            "label": 1,
            **features
        })

    except Exception as e:
        print(f"Failed: {file} | {e}")

# =========================================================
# SAVE
# =========================================================

final_df = pd.DataFrame(all_rows)

final_df.drop_duplicates(subset=["text"], inplace=True)

os.makedirs("./data", exist_ok=True)

output_path = "./data/feature_dataset.csv"

final_df.to_csv(output_path, index=False)

print("\n[+] FINAL DATASET CREATED")
print(final_df.head())

print("\nShape:")
print(final_df.shape)

print(f"\nSaved to: {output_path}")