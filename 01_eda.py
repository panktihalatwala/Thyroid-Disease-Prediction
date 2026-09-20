"""
STEP 1b: Exploratory Data Analysis (EDA)
Loads data/thyroid_raw.csv (created by 01_load_data.py) and produces
summary stats and plots saved into the eda_outputs/ folder.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # so it saves files instead of trying to pop up a window
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = "data/thyroid_raw.csv"
OUT_DIR = "eda_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print("\nColumn dtypes:\n", df.dtypes)

# ---- Basic stats ----
print("\nDescribe (numeric columns):")
numeric_cols = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
print(df[numeric_cols].describe())

# ---- Missing values ----
print("\nMissing values per column:")
missing = df.isna().sum()
print(missing[missing > 0])

# ---- Target distribution ----
print("\nTarget distribution:")
print(df["target"].value_counts())
print(df["target"].value_counts(normalize=True))

plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="target")
plt.title("Target Class Distribution")
plt.savefig(f"{OUT_DIR}/target_distribution.png", bbox_inches="tight")
plt.close()

# ---- Numeric feature distributions by target ----
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=col, hue="target", kde=True, element="step")
    plt.title(f"{col} distribution by target")
    plt.savefig(f"{OUT_DIR}/{col}_distribution.png", bbox_inches="tight")
    plt.close()

# ---- Correlation heatmap (numeric features only) ----
plt.figure(figsize=(7, 5))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Numeric Features")
plt.savefig(f"{OUT_DIR}/correlation_heatmap.png", bbox_inches="tight")
plt.close()

# ---- Age distribution overall ----
plt.figure(figsize=(6, 4))
sns.histplot(df["age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.savefig(f"{OUT_DIR}/age_distribution.png", bbox_inches="tight")
plt.close()

print(f"\nDone. Plots saved in {OUT_DIR}/")