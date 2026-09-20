"""
STEP 3: Baseline Model Training and Comparison
Loads processed_train.csv and processed_test.csv, trains 5 models,
and compares them using accuracy, precision, recall, F1, and confusion matrix.

Note: Logistic Regression and SVM are scale-sensitive, so we apply
StandardScaler for those two. Decision Tree, Random Forest, and XGBoost
are tree-based and don't need scaling, so they use the raw features.
This keeps the comparison fair to every algorithm.

Saves trained models to models/ folder for later use (feature selection, SHAP).
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import StandardScaler
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
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

# ---- Load data ----
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]
X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

# ---- Create a scaled version for scale-sensitive models ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler too - needed later if you deploy the Streamlit app
joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")

# ---- Define models: (model, needs_scaling) ----
models = {
    "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=42), True),
    "Decision Tree": (DecisionTreeClassifier(random_state=42), False),
    "Random Forest": (RandomForestClassifier(random_state=42), False),
    "SVM": (SVC(probability=True, random_state=42), True),
    "XGBoost": (XGBClassifier(random_state=42, eval_metric="logloss"), False),
}

results = []

for name, (model, needs_scaling) in models.items():
    print(f"\n{'='*50}\nTraining: {name}\n{'='*50}")

    if needs_scaling:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "Scaled": needs_scaling,
    })

    safe_name = name.replace(" ", "_").lower()
    joblib.dump(model, f"{MODELS_DIR}/{safe_name}.pkl")

# ---- Comparison table ----
results_df = pd.DataFrame(results).sort_values(by="F1-score", ascending=False)
print(f"\n{'='*50}\nMODEL COMPARISON (sorted by F1-score)\n{'='*50}")
print(results_df.to_string(index=False))

results_df.to_csv("results/model_comparison.csv", index=False)
print("\nSaved comparison table to results/model_comparison.csv")
print(f"Saved trained models to {MODELS_DIR}/")