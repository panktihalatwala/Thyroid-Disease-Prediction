"""
STEP 2: Preprocessing
Owner: PANKTI
Run this AFTER 01_eda.py, once you know the real column names and target column.
Output: data/processed_train.csv and data/processed_test.csv
These two files are what everyone else (baseline models, feature selection, SHAP)
will load - so once this script runs correctly, push it to GitHub immediately
so Shreena can pull the same processed data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---- EDIT THESE only if you rename things above ----
DATA_PATH = "data/thyroid_raw.csv"
TARGET_COL = "target"                       # binary: "negative" / "sick"
DROP_COLS = ["TBG", "TBG_measured", "diagnosis", "patient_id"]

df = pd.read_csv(DATA_PATH)

# 1. Drop columns that are mostly missing / not useful (TBG is ~95% missing)
#    and the raw diagnosis/id columns we no longer need
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# 2. Replace any leftover '?' missing-value markers with real NaN
df = df.replace("?", np.nan)

# 3. Separate target from features
y_raw = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

# 4. Split columns into numeric vs categorical automatically
#    (after replacing '?', numeric columns are still stored as text - fix that)
likely_numeric = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
for col in likely_numeric:
    if col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# 5. Impute missing values
#    Numeric -> median | Categorical -> most frequent value
for col in numeric_cols:
    X[col] = X[col].fillna(X[col].median())

for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

# 6. Encode categorical columns
#    Binary t/f or M/F columns -> 0/1 ; multi-category columns -> one-hot
binary_like_cols = [c for c in categorical_cols if X[c].nunique() == 2]
multi_cat_cols = [c for c in categorical_cols if X[c].nunique() > 2]

for col in binary_like_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

if multi_cat_cols:
    X = pd.get_dummies(X, columns=multi_cat_cols, drop_first=True)

# 7. Encode the target
target_le = LabelEncoder()
y = target_le.fit_transform(y_raw)
print("\nTarget classes:", dict(zip(target_le.classes_, range(len(target_le.classes_)))))

# 8. Train/test split (stratified because the classes are imbalanced)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain shape: {X_train.shape} | Test shape: {X_test.shape}")
print("Train class balance:\n", pd.Series(y_train).value_counts(normalize=True))

# 9. Save processed data so everyone works off the SAME preprocessed files
X_train.assign(target=y_train).to_csv("data/processed_train.csv", index=False)
X_test.assign(target=y_test).to_csv("data/processed_test.csv", index=False)

print("\nSaved: data/processed_train.csv and data/processed_test.csv")
print("Commit and push these two files (or the script + raw data) so Shreena can pull them.")