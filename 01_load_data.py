"""
STEP 1 (rewritten): Load and parse thyroid0387.data exactly.
This replaces the earlier ucimlrepo-based version, since that dataset
isn't available through the API - we're using the manually downloaded
file instead.

HOW TO USE:
1. Put thyroid0387.data in a folder called data/ next to this script.
2. Run: python 01_load_data.py
It will save a clean CSV to data/thyroid_raw.csv with proper column
names and a simplified binary target, ready for 02_preprocessing.py.
"""

import pandas as pd
import numpy as np

DATA_PATH = "data/thyroid0387.data"

# Exact column order, confirmed from thyroid0387.names
COLUMNS = [
    "age", "sex", "on_thyroxine", "query_on_thyroxine",
    "on_antithyroid_medication", "sick", "pregnant", "thyroid_surgery",
    "I131_treatment", "query_hypothyroid", "query_hyperthyroid",
    "lithium", "goitre", "tumor", "hypopituitary", "psych",
    "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured", "TT4",
    "T4U_measured", "T4U", "FTI_measured", "FTI", "TBG_measured", "TBG",
    "referral_source",
]

# ---- Read the raw file ----
# The file has 29 comma-separated attributes, then the 30th "column" is
# the diagnosis string glued directly to "[patient_id]" with no comma,
# e.g. "-[840801013]" or "AK[850906002]". We read 29 fixed columns and
# capture the rest as one raw string, then split it ourselves.
raw = pd.read_csv(DATA_PATH, header=None, names=COLUMNS + ["diag_raw"])

# Split "diag_raw" into diagnosis codes and patient id, e.g. "AK[850906002]"
split = raw["diag_raw"].str.extract(r"^([^\[]*)\[(\d+)\]$")
raw["diagnosis"] = split[0]
raw["patient_id"] = split[1]
raw = raw.drop(columns=["diag_raw"])

# ---- Replace missing-value marker ----
raw = raw.replace("?", np.nan)

# ---- Build a simple binary target ----
# "-" (or a discordant "X|Y" reading of only R/S/T that don't indicate a
# real condition) means no condition flagged; anything else means at
# least one condition was flagged. This mirrors how "sick.data" in the
# same archive simplifies the full diagnosis string.
raw["target"] = raw["diagnosis"].apply(lambda d: "negative" if d.strip() == "-" else "sick")

print("Shape:", raw.shape)
print("\nTarget distribution:")
print(raw["target"].value_counts())
print("\nMissing values per column:")
print(raw.isna().sum()[raw.isna().sum() > 0])

raw.to_csv("data/thyroid_raw.csv", index=False)
print("\nSaved data/thyroid_raw.csv - ready for 02_preprocessing.py")