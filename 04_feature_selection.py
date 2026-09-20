"""
STEP 4: Feature Selection (multi-class). RFE + Mutual Information both
work natively with multi-class targets. XGBoost again uses
compute_sample_weight since it has no class_weight parameter.
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

TRAIN_PATH = "data/processed_train.csv"
TEST_PATH = "data/processed_test.csv"
RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_FEATURES_TO_SELECT = 15

with open("data/target_classes.txt") as f:
    class_names = [line.strip() for line in f if line.strip()]

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]
X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

print(f"Full feature set: {X_train.shape[1]} features")

# ---- RFE ----
print(f"\n{'='*60}\nMETHOD 1: RFE (Random Forest)\n{'='*60}")
rfe_estimator = RandomForestClassifier(random_state=42, class_weight="balanced")
rfe = RFE(estimator=rfe_estimator, n_features_to_select=N_FEATURES_TO_SELECT)
rfe.fit(X_train, y_train)
rfe_selected_features = X_train.columns[rfe.support_].tolist()
print("RFE selected:", rfe_selected_features)

# ---- Mutual Information ----
print(f"\n{'='*60}\nMETHOD 2: Mutual Information\n{'='*60}")
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_df = pd.DataFrame({"Feature": X_train.columns, "MI_Score": mi_scores}).sort_values(
    by="MI_Score", ascending=False
)
print(mi_df.to_string(index=False))
mi_selected_features = mi_df.head(N_FEATURES_TO_SELECT)["Feature"].tolist()
mi_df.to_csv(f"{RESULTS_DIR}/mutual_information_scores.csv", index=False)

overlap = set(rfe_selected_features) & set(mi_selected_features)
print(f"\nOverlap ({len(overlap)}): {sorted(overlap)}")
print(f"Only in RFE: {sorted(set(rfe_selected_features) - overlap)}")
print(f"Only in Mutual Information: {sorted(set(mi_selected_features) - overlap)}")

# ---- Retrain on RFE-selected features ----
print(f"\n{'='*60}\nRETRAINING ON RFE-SELECTED FEATURES\n{'='*60}")

X_train_sel = X_train[rfe_selected_features]
X_test_sel = X_test[rfe_selected_features]

scaler = StandardScaler()
X_train_sel_scaled = scaler.fit_transform(X_train_sel)
X_test_sel_scaled = scaler.transform(X_test_sel)
joblib.dump(scaler, f"{MODELS_DIR}/scaler_selected.pkl")

models = {
    "Logistic Regression": (LogisticRegression(max_iter=3000, random_state=42, class_weight="balanced"), True),
    "Decision Tree": (DecisionTreeClassifier(random_state=42, class_weight="balanced"), False),
    "Random Forest": (RandomForestClassifier(random_state=42, class_weight="balanced"), False),
    "SVM": (SVC(probability=True, random_state=42, class_weight="balanced"), True),
    "XGBoost": (XGBClassifier(random_state=42, eval_metric="mlogloss"), False),
}

results = []
for name, (model, needs_scaling) in models.items():
    X_tr = X_train_sel_scaled if needs_scaling else X_train_sel
    X_te = X_test_sel_scaled if needs_scaling else X_test_sel

    if name == "XGBoost":
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(X_tr, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n{name}: Acc={acc:.4f} Prec(macro)={prec:.4f} Rec(macro)={rec:.4f} F1(macro)={f1:.4f}")

    results.append({"Model": name, "Accuracy": acc, "Precision(macro)": prec,
                     "Recall(macro)": rec, "F1(macro)": f1})

    safe_name = name.replace(" ", "_").lower()
    joblib.dump(model, f"{MODELS_DIR}/{safe_name}_selected_features.pkl")

results_df = pd.DataFrame(results).sort_values(by="F1(macro)", ascending=False)
print(f"\n{'='*60}\nMODEL COMPARISON - SELECTED FEATURES\n{'='*60}")
print(results_df.to_string(index=False))
results_df.to_csv(f"{RESULTS_DIR}/model_comparison_selected_features.csv", index=False)

with open(f"{RESULTS_DIR}/selected_features.txt", "w") as f:
    f.write("\n".join(rfe_selected_features))

print(f"\nSaved everything to {RESULTS_DIR}/ and {MODELS_DIR}/")