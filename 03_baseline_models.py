"""
STEP 3: Baseline Model Training and Comparison (multi-class).

Uses macro-averaged precision/recall/F1 (treats every class equally,
important since classes are imbalanced) alongside a full per-class
classification report.

XGBoost doesn't support class_weight="balanced" directly, so we use
compute_sample_weight to achieve the same effect via sample_weight
during training - keeping the comparison fair across all 5 models.
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

TRAIN_PATH = "data/processed_train.csv"
TEST_PATH = "data/processed_test.csv"
MODELS_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

with open("data/target_classes.txt") as f:
    class_names = [line.strip() for line in f if line.strip()]
print("Classes (in label order):", class_names)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]
X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")

models = {
    "Logistic Regression": (LogisticRegression(max_iter=3000, random_state=42, class_weight="balanced"), True),
    "Decision Tree": (DecisionTreeClassifier(random_state=42, class_weight="balanced"), False),
    "Random Forest": (RandomForestClassifier(random_state=42, class_weight="balanced"), False),
    "SVM": (SVC(probability=True, random_state=42, class_weight="balanced"), True),
    "XGBoost": (XGBClassifier(random_state=42, eval_metric="mlogloss"), False),
}

results = []

for name, (model, needs_scaling) in models.items():
    print(f"\n{'='*60}\nTraining: {name}\n{'='*60}")

    X_tr = X_train_scaled if needs_scaling else X_train
    X_te = X_test_scaled if needs_scaling else X_test

    if name == "XGBoost":
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(X_tr, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy:        {acc:.4f}")
    print(f"Precision(macro):{prec_macro:.4f}")
    print(f"Recall(macro):   {rec_macro:.4f}")
    print(f"F1(macro):       {f1_macro:.4f}")
    print(f"F1(weighted):    {f1_weighted:.4f}")
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print("Classes order:", class_names)
    print(cm)
    print("\nClassification Report:\n",
          classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision(macro)": prec_macro,
        "Recall(macro)": rec_macro,
        "F1(macro)": f1_macro,
        "F1(weighted)": f1_weighted,
    })

    safe_name = name.replace(" ", "_").lower()
    joblib.dump(model, f"{MODELS_DIR}/{safe_name}.pkl")

results_df = pd.DataFrame(results).sort_values(by="F1(macro)", ascending=False)
print(f"\n{'='*60}\nMODEL COMPARISON (sorted by macro F1-score)\n{'='*60}")
print(results_df.to_string(index=False))

results_df.to_csv(f"{RESULTS_DIR}/model_comparison.csv", index=False)
print(f"\nSaved comparison table to {RESULTS_DIR}/model_comparison.csv")
print(f"Saved trained models to {MODELS_DIR}/")