# =========================================================
# dataset_analysis.py
# TOOL C DATASET + LEXICON ANALYTICS
# =========================================================

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

from wordcloud import WordCloud

# =========================================================
# PATHS
# =========================================================

FEATURE_DATASET_PATH = "./data/feature_dataset.csv"

LEXICON_PATH = "./data/scam_lexicon.csv"

OUTPUT_DIR = "./dataset_graphs"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# =========================================================
# LOAD DATASETS
# =========================================================

print("\n[+] Loading datasets...")

df = pd.read_csv(
    FEATURE_DATASET_PATH
)

lexicon_df = pd.read_csv(
    LEXICON_PATH
)

print("\nFeature Dataset Shape:")
print(df.shape)

print("\nLexicon Shape:")
print(lexicon_df.shape)

# =========================================================
# CLEAN DATA
# =========================================================

df = df.dropna(
    subset=["text", "label"]
)

df["text"] = (
    df["text"]
    .astype(str)
)

# =========================================================
# LABEL DISTRIBUTION
# =========================================================

print("\n[+] Computing label distribution...")

label_counts = (
    df["label"]
    .value_counts()
)

plt.figure(figsize=(7, 6))

plt.bar(

    ["HAM", "SCAM"],

    [
        label_counts.get(0, 0),
        label_counts.get(1, 0)
    ]
)

plt.ylabel("Count")

plt.title(
    "Tool C Dataset Class Distribution"
)

for i, v in enumerate([

    label_counts.get(0, 0),

    label_counts.get(1, 0)

]):

    plt.text(
        i,
        v + 20,
        str(v),
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/class_distribution.png"
)

plt.close()

# =========================================================
# LEXICON CATEGORY DISTRIBUTION
# =========================================================

print("\n[+] Building category distribution...")

if "category" in lexicon_df.columns:

    category_counts = (

        lexicon_df["category"]
        .value_counts()
    )

    plt.figure(figsize=(10, 6))

    plt.bar(

        category_counts.index,

        category_counts.values
    )

    plt.xticks(rotation=25)

    plt.ylabel("Phrase Count")

    plt.title(
        "Scam Lexicon Category Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/category_distribution.png"
    )

    plt.close()

# =========================================================
# TOP SCAM PHRASES
# =========================================================

print("\n[+] Finding top scam phrases...")

if "spam_count" in lexicon_df.columns:

    top_phrases = (

        lexicon_df
        .sort_values(
            by="spam_count",
            ascending=False
        )
        .head(15)
    )

    plt.figure(figsize=(12, 7))

    plt.barh(

        top_phrases["phrase"],

        top_phrases["spam_count"]
    )

    plt.xlabel("Frequency")

    plt.title(
        "Top Scam Phrases"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/top_scam_phrases.png"
    )

    plt.close()

# =========================================================
# WORD CLOUD — SCAM MESSAGES
# =========================================================

print("\n[+] Building scam word cloud...")

scam_text = " ".join(

    df[df["label"] == 1]["text"]
    .tolist()
)

wordcloud = WordCloud(

    width=1200,

    height=600,

    background_color="white"
).generate(scam_text)

plt.figure(figsize=(14, 7))

plt.imshow(
    wordcloud,
    interpolation="bilinear"
)

plt.axis("off")

plt.title(
    "Scam Message Word Cloud"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/scam_wordcloud.png"
)

plt.close()

# =========================================================
# WORD CLOUD — NORMAL MESSAGES
# =========================================================

print("\n[+] Building normal word cloud...")

ham_text = " ".join(

    df[df["label"] == 0]["text"]
    .tolist()
)

wordcloud = WordCloud(

    width=1200,

    height=600,

    background_color="white"
).generate(ham_text)

plt.figure(figsize=(14, 7))

plt.imshow(
    wordcloud,
    interpolation="bilinear"
)

plt.axis("off")

plt.title(
    "Normal Message Word Cloud"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/ham_wordcloud.png"
)

plt.close()

# =========================================================
# MESSAGE LENGTH DISTRIBUTION
# =========================================================

print("\n[+] Analyzing message lengths...")

df["message_length"] = (

    df["text"]
    .apply(len)
)

scam_lengths = (

    df[df["label"] == 1]
    ["message_length"]
)

ham_lengths = (

    df[df["label"] == 0]
    ["message_length"]
)

plt.figure(figsize=(12, 6))

plt.hist(

    ham_lengths,

    bins=40,

    alpha=0.6,

    label="HAM"
)

plt.hist(

    scam_lengths,

    bins=40,

    alpha=0.6,

    label="SCAM"
)

plt.xlabel("Message Length")

plt.ylabel("Frequency")

plt.title(
    "Message Length Distribution"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/message_length_distribution.png"
)

plt.close()

# =========================================================
# LEXICON STATISTICS
# =========================================================

print("\n[+] Building lexicon statistics...")

stats = {

    "Total Messages":
        len(df),

    "Scam Messages":
        len(df[df["label"] == 1]),

    "Ham Messages":
        len(df[df["label"] == 0]),

    "Lexicon Size":
        len(lexicon_df),

    "Unique Categories":
        lexicon_df["category"].nunique()
        if "category" in lexicon_df.columns
        else 0,
}

stats_df = pd.DataFrame(

    list(stats.items()),

    columns=["Metric", "Value"]
)

stats_df.to_csv(

    f"{OUTPUT_DIR}/dataset_statistics.csv",

    index=False
)

# =========================================================
# VISUALIZE DATASET STATISTICS
# =========================================================

plt.figure(figsize=(10, 6))

plt.bar(

    stats_df["Metric"],

    stats_df["Value"]
)

plt.xticks(rotation=20)

plt.title(
    "Tool C Dataset Statistics"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/dataset_statistics.png"
)

plt.close()

# =========================================================
# AVERAGE MESSAGE LENGTHS
# =========================================================

avg_scam_length = round(
    scam_lengths.mean(),
    2
)

avg_ham_length = round(
    ham_lengths.mean(),
    2
)

print("\nAverage Scam Message Length:")
print(avg_scam_length)

print("\nAverage Ham Message Length:")
print(avg_ham_length)

# =========================================================
# SAVE SUMMARY REPORT
# =========================================================

summary_path = (
    f"{OUTPUT_DIR}/summary_report.txt"
)

with open(summary_path, "w") as f:

    f.write(
        "TOOL C DATASET ANALYSIS REPORT\n"
    )

    f.write(
        "=================================\n\n"
    )

    for k, v in stats.items():

        f.write(f"{k}: {v}\n")

    f.write("\n")

    f.write(
        f"Average Scam Length: "
        f"{avg_scam_length}\n"
    )

    f.write(
        f"Average Ham Length: "
        f"{avg_ham_length}\n"
    )

print("\n[+] Analysis complete.")

print("\nGraphs saved to:")
print(OUTPUT_DIR)

print("\nGenerated Files:")

generated_files = os.listdir(
    OUTPUT_DIR
)

for file in generated_files:

    print(f" - {file}")