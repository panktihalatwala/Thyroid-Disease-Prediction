"""
STEP 1: Exploratory Data Analysis (EDA)
Run this FIRST, before writing any preprocessing code.
Purpose: understand columns, missing values, class balance.

HOW TO USE:
This uses the official `ucimlrepo` package to pull the UCI Thyroid Disease
dataset (id=102) directly - no manual file download needed.
Just run: pip install ucimlrepo   (already in requirements.txt)
Then:     python 01_eda.py
It will also save a local copy to data/thyroid_raw.csv so you're not
re-downloading it every time.
"""

import pandas as pd
from ucimlrepo import fetch_ucirepo

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

thyroid = fetch_ucirepo(id=102)
X_raw = thyroid.data.features
y_raw = thyroid.data.targets

df = pd.concat([X_raw, y_raw], axis=1)
df.to_csv("data/thyroid_raw.csv", index=False)
print("Saved a local copy to data/thyroid_raw.csv\n")

print("METADATA:")
print(thyroid.metadata.get("abstract", ""))
print("\nVARIABLE INFO:")
print(thyroid.variables)

print("=" * 60)
print("SHAPE:", df.shape)
print("=" * 60)

print("\nCOLUMN NAMES:")
print(list(df.columns))

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nDATA TYPES:")
print(df.dtypes)

print("\nMISSING VALUES (as '?' - common in this dataset):")
question_mark_counts = (df == "?").sum()
print(question_mark_counts[question_mark_counts > 0])

print("\nMISSING VALUES (actual NaN, if any):")
print(df.isna().sum()[df.isna().sum() > 0])

target_col = y_raw.columns[0]
print(f"\nTarget column: '{target_col}'")

print(f"\nCLASS DISTRIBUTION ('{target_col}'):")
print(df[target_col].value_counts())
print("\nClass distribution (%):")
print(df[target_col].value_counts(normalize=True) * 100)

print("\nNUMERIC COLUMN SUMMARY (may need cleaning first if stored as text):")
print(df.describe(include="all").T)

print("\n" + "=" * 60)
print("NEXT STEP: note down the exact target column name and which")
print("columns are mostly missing (candidates to drop, e.g. TBG).")
print("Then move to 02_preprocessing.py")
print("=" * 60)