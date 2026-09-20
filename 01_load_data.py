"""
STEP 1: Load and parse thyroid0387.data, with a 4-CLASS target.

Classes:
  - negative     : no condition flagged
  - hyperthyroid : diagnosis codes A, B, C, D
  - hypothyroid  : diagnosis codes E, F, G, H
  - other        : any other flagged condition (binding protein issues,
                    replacement therapy status, miscellaneous findings)

HOW TO USE:
1. Put thyroid0387.data in a folder called data/ next to this script.
2. Run: python 01_load_data.py
It saves data/thyroid_raw.csv, ready for 02_preprocessing.py.
"""

import pandas as pd
import numpy as np

DATA_PATH = "data/thyroid0387.data"

COLUMNS = [
    "age", "sex", "on_thyroxine", "query_on_thyroxine",
    "on_antithyroid_medication", "sick", "pregnant", "thyroid_surgery",
    "I131_treatment", "query_hypothyroid", "query_hyperthyroid",
    "lithium", "goitre", "tumor", "hypopituitary", "psych",
    "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured", "TT4",
    "T4U_measured", "T4U", "FTI_measured", "FTI", "TBG_measured", "TBG",
    "referral_source",
]

raw = pd.read_csv(DATA_PATH, header=None, names=COLUMNS + ["diag_raw"])

split = raw["diag_raw"].str.extract(r"^([^\[]*)\[(\d+)\]$")
raw["diagnosis"] = split[0]
raw["patient_id"] = split[1]
raw = raw.drop(columns=["diag_raw"])

raw = raw.replace("?", np.nan)

# ---- Multi-class target ----
HYPERTHYROID_CODES = set("ABCD")
HYPOTHYROID_CODES = set("EFGH")

def classify_diagnosis(diag):
    diag = diag.strip()
    if diag == "-":
        return "negative"
    letters = set(diag) - {"-"}
    if letters & HYPERTHYROID_CODES:
        return "hyperthyroid"
    if letters & HYPOTHYROID_CODES:
        return "hypothyroid"
    return "other"

raw["target"] = raw["diagnosis"].apply(classify_diagnosis)

print("Shape:", raw.shape)
print("\nTarget distribution:")
print(raw["target"].value_counts())
print(raw["target"].value_counts(normalize=True))

print("\nMissing values per column:")
print(raw.isna().sum()[raw.isna().sum() > 0])

raw.to_csv("data/thyroid_raw.csv", index=False)
print("\nSaved data/thyroid_raw.csv - ready for 02_preprocessing.py")