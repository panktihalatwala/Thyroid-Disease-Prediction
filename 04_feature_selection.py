"""
STEP 4: Feature Selection
Uses two techniques to identify the most important clinical features:
  1. RFE (Recursive Feature Elimination) - using Random Forest as the estimator
  2. Mutual Information - measures dependency between each feature and the target

Then retrains all 5 models on the reduced feature set so we can compare
"all features" vs "selected features" performance - this is the core
of the research contribution.
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

TRAIN_PATH = "data/processed_train.csv"
TEST_PATH = "data/processed_test.csv"
RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_FEATURES_TO_SELECT = 15  # adjust if you want more/fewer features

# ---- Load data ----
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]
X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

print(f"Full feature set: {X_train.shape[1]} features")
print(list(X_train.columns))

# ============================================================
# METHOD 1: RFE (Recursive Feature Elimination)
# ============================================================
print(f"\n{'='*60}\nMETHOD 1: RFE (using Random Forest)\n{'='*60}")

rfe_estimator = RandomForestClassifier(random_state=42)
rfe = RFE(estimator=rfe_estimator, n_features_to_select=N_FEATURES_TO_SELECT)
rfe.fit(X_train, y_train)

rfe_selected_features = X_train.columns[rfe.support_].tolist()
print(f"\nRFE selected {len(rfe_selected_features)} features:")
print(rfe_selected_features)

# ============================================================
# METHOD 2: Mutual Information
# ============================================================
print(f"\n{'='*60}\nMETHOD 2: Mutual Information\n{'='*60}")

mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_df = pd.DataFrame({
    "Feature": X_train.columns,
    "MI_Score": mi_scores
}).sort_values(by="MI_Score", ascending=False)

print("\nAll features ranked by Mutual Information score:")
print(mi_df.to_string(index=False))

mi_selected_features = mi_df.head(N_FEATURES_TO_SELECT)["Feature"].tolist()
print(f"\nTop {N_FEATURES_TO_SELECT} features by Mutual Information:")
print(mi_selected_features)

mi_df.to_csv(f"{RESULTS_DIR}/mutual_information_scores.csv", index=False)

# ============================================================
# Compare overlap between the two methods
# ============================================================
overlap = set(rfe_selected_features) & set(mi_selected_features)
print(f"\n{'='*60}\nOverlap between RFE and Mutual Information\n{'='*60}")
print(f"Features selected by BOTH methods ({len(overlap)}): {sorted(overlap)}")
print(f"Only in RFE: {sorted(set(rfe_selected_features) - overlap)}")
print(f"Only in Mutual Information: {sorted(set(mi_selected_features) - overlap)}")

# ============================================================
# Retrain all 5 models using RFE-selected features
# (RFE is used here since it's more commonly reported as the final
# selected set in similar papers - Mutual Information is shown for comparison)
# ============================================================
print(f"\n{'='*60}\nRETRAINING MODELS ON RFE-SELECTED FEATURES\n{'='*60}")

X_train_sel = X_train[rfe_selected_features]
X_test_sel = X_test[rfe_selected_features]

scaler = StandardScaler()
X_train_sel_scaled = scaler.fit_transform(X_train_sel)
X_test_sel_scaled = scaler.transform(X_test_sel)

models = {
    "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=42), True),
    "Decision Tree": (DecisionTreeClassifier(random_state=42), False),
    "Random Forest": (RandomForestClassifier(random_state=42), False),
    "SVM": (SVC(probability=True, random_state=42), True),
    "XGBoost": (XGBClassifier(random_state=42, eval_metric="logloss"), False),
}

results = []

for name, (model, needs_scaling) in models.items():
    X_tr = X_train_sel_scaled if needs_scaling else X_train_sel
    X_te = X_test_sel_scaled if needs_scaling else X_test_sel

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{name}:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-score:  {f1:.4f}")

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
    })

    safe_name = name.replace(" ", "_").lower()
    joblib.dump(model, f"{MODELS_DIR}/{safe_name}_selected_features.pkl")

results_df = pd.DataFrame(results).sort_values(by="F1-score", ascending=False)
print(f"\n{'='*60}\nMODEL COMPARISON - SELECTED FEATURES ONLY\n{'='*60}")
print(results_df.to_string(index=False))

results_df.to_csv(f"{RESULTS_DIR}/model_comparison_selected_features.csv", index=False)

# Save the selected feature list for later use (SHAP, Streamlit app)
with open(f"{RESULTS_DIR}/selected_features.txt", "w") as f:
    f.write("\n".join(rfe_selected_features))

print(f"\nSaved comparison table to {RESULTS_DIR}/model_comparison_selected_features.csv")
print(f"Saved selected feature list to {RESULTS_DIR}/selected_features.txt")
print(f"Saved models trained on selected features to {MODELS_DIR}/")