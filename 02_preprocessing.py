"""
STEP 2: Preprocessing (multi-class target: negative / hyperthyroid /
hypothyroid / other). Same cleaning logic as before - only the target
now has 4 classes instead of 2.

Output: data/processed_train.csv and data/processed_test.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/thyroid_raw.csv"
TARGET_COL = "target"
DROP_COLS = ["TBG", "TBG_measured", "diagnosis", "patient_id"]

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
df = df.replace("?", np.nan)

y_raw = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

likely_numeric = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
for col in likely_numeric:
    if col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

for col in numeric_cols:
    X[col] = X[col].fillna(X[col].median())
for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

binary_like_cols = [c for c in categorical_cols if X[c].nunique() == 2]
multi_cat_cols = [c for c in categorical_cols if X[c].nunique() > 2]

for col in binary_like_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

if multi_cat_cols:
    X = pd.get_dummies(X, columns=multi_cat_cols, drop_first=True)

target_le = LabelEncoder()
y = target_le.fit_transform(y_raw)

print("\nTarget classes:", dict(zip(target_le.classes_, range(len(target_le.classes_)))))

# Save the class name order - every later script needs this to map
# numeric predictions back to readable class names
with open("data/target_classes.txt", "w") as f:
    f.write("\n".join(target_le.classes_))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain shape: {X_train.shape} | Test shape: {X_test.shape}")
print("Train class balance:\n", pd.Series(y_train).value_counts(normalize=True))

X_train.assign(target=y_train).to_csv("data/processed_train.csv", index=False)
X_test.assign(target=y_test).to_csv("data/processed_test.csv", index=False)

print("\nSaved: data/processed_train.csv, data/processed_test.csv, data/target_classes.txt")