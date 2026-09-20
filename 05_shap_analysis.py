"""
STEP 5: SHAP Explainability (multi-class) + plain-English explanations.

For multi-class models, SHAP produces one set of values PER CLASS
(how much each feature pushed toward THAT specific class). We generate:
  1. A summary/bar plot per class
  2. A feature importance table per class
  3. Waterfall plots for one example per class
  4. A human-readable sentence explaining each example prediction,
     built from the actual SHAP values and how each value compares
     to the training data's typical range.
"""

import pandas as pd
import numpy as np
import joblib
import shap
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN_PATH = "data/processed_train.csv"
TEST_PATH = "data/processed_test.csv"
MODELS_DIR = "models"
OUT_DIR = "shap_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

with open("results/selected_features.txt") as f:
    selected_features = [line.strip() for line in f if line.strip()]
with open("data/target_classes.txt") as f:
    class_names = [line.strip() for line in f if line.strip()]

print("Selected features:", selected_features)
print("Classes:", class_names)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df[selected_features]
y_train = train_df["target"]
X_test = test_df[selected_features]
y_test = test_df["target"]

model = joblib.load(f"{MODELS_DIR}/xgboost_selected_features.pkl")
print("\nLoaded trained XGBoost model.")

# ---- Compute reference stats (median) from TRAINING data, used to say
# whether a value is "higher/lower than typical" - not a medical claim,
# just a comparison to this dataset's own distribution ----
feature_medians = X_train.median()

print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(model)
explanation = explainer(X_test)   # new-style Explanation object, multi-output aware

# explanation.values shape: (n_samples, n_features, n_classes) for multiclass
n_classes = explanation.values.shape[2]
print(f"SHAP values computed for {n_classes} classes.")

# ============================================================
# 1 & 2: Summary plots + importance table, per class
# ============================================================
all_importance_rows = []

for class_idx, class_name in enumerate(class_names):
    class_shap_values = explanation.values[:, :, class_idx]

    plt.figure()
    shap.summary_plot(class_shap_values, X_test, show=False)
    plt.title(f"SHAP Summary - Class: {class_name}")
    plt.savefig(f"{OUT_DIR}/shap_summary_{class_name}.png", bbox_inches="tight", dpi=150)
    plt.close()

    plt.figure()
    shap.summary_plot(class_shap_values, X_test, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance - Class: {class_name}")
    plt.savefig(f"{OUT_DIR}/shap_bar_{class_name}.png", bbox_inches="tight", dpi=150)
    plt.close()

    mean_abs = np.abs(class_shap_values).mean(axis=0)
    for feat, val in zip(selected_features, mean_abs):
        all_importance_rows.append({"Class": class_name, "Feature": feat, "Mean_Abs_SHAP": val})

    print(f"Saved plots for class: {class_name}")

importance_df = pd.DataFrame(all_importance_rows)
importance_df.to_csv(f"{OUT_DIR}/shap_importance_by_class.csv", index=False)
print(f"\nSaved: {OUT_DIR}/shap_importance_by_class.csv")

print("\nTop 5 features per class:")
for class_name in class_names:
    top5 = importance_df[importance_df["Class"] == class_name].sort_values(
        by="Mean_Abs_SHAP", ascending=False
    ).head(5)
    print(f"\n{class_name}:")
    print(top5[["Feature", "Mean_Abs_SHAP"]].to_string(index=False))

# ============================================================
# 3 & 4: Waterfall + plain-English explanation, one example per class
# ============================================================
y_pred = model.predict(X_test)

def explain_in_words(row_idx, predicted_class_idx, predicted_class_name):
    """Builds a plain-English sentence from the top 3 SHAP contributors
    for this specific prediction."""
    shap_row = explanation.values[row_idx, :, predicted_class_idx]
    feature_values = X_test.iloc[row_idx]

    # Sort features by absolute impact, take top 3
    impact_order = np.argsort(-np.abs(shap_row))[:3]

    phrases = []
    for i in impact_order:
        feat = selected_features[i]
        impact = shap_row[i]
        value = feature_values[feat]
        median = feature_medians[feat]

        direction = "pushed TOWARD" if impact > 0 else "pushed AWAY FROM"
        level = "higher than typical" if value > median else "lower than typical" if value < median else "close to typical"

        phrases.append(f"{feat} = {value:.2f} ({level} for this dataset) {direction} '{predicted_class_name}'")

    sentence = (
        f"This case was predicted as '{predicted_class_name}'. "
        f"The strongest factors were: " + "; ".join(phrases) + "."
    )
    return sentence

print(f"\n{'='*60}\nEXAMPLE EXPLANATIONS (one per class)\n{'='*60}")

explanation_log = []

for class_idx, class_name in enumerate(class_names):
    # Find a correctly-predicted example for this class
    match_idx = None
    for i in range(len(y_test)):
        if y_test.iloc[i] == class_idx and y_pred[i] == class_idx:
            match_idx = i
            break

    if match_idx is None:
        print(f"\nNo correctly predicted example found for class '{class_name}' - skipping.")
        continue

    sentence = explain_in_words(match_idx, class_idx, class_name)
    print(f"\n{sentence}")
    explanation_log.append({"Class": class_name, "Explanation": sentence})

    # Waterfall plot for this example, for the predicted class
    plt.figure()
    single_explanation = shap.Explanation(
        values=explanation.values[match_idx, :, class_idx],
        base_values=explanation.base_values[match_idx, class_idx],
        data=explanation.data[match_idx],
        feature_names=selected_features
    )
    shap.plots.waterfall(single_explanation, show=False)
    plt.title(f"SHAP Waterfall - Example: {class_name}")
    plt.savefig(f"{OUT_DIR}/shap_waterfall_{class_name}.png", bbox_inches="tight", dpi=150)
    plt.close()

pd.DataFrame(explanation_log).to_csv(f"{OUT_DIR}/example_explanations.csv", index=False)
print(f"\nSaved example explanations to {OUT_DIR}/example_explanations.csv")
print(f"All SHAP outputs saved in {OUT_DIR}/")