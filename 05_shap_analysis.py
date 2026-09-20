"""
STEP 5: SHAP Explainability
Loads the XGBoost model trained on RFE-selected features (from Step 4)
and generates SHAP explanations:
  1. Summary plot - which features matter most overall, and how
  2. Bar plot - mean absolute SHAP value per feature (simple importance ranking)
  3. Waterfall plots for a few individual predictions - shows exactly why
     the model predicted "sick" or "negative" for that specific patient

Outputs are saved as .png files in shap_outputs/ for use in the
research paper and the Streamlit prototype.
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

# ---- Load selected feature list (from Step 4) ----
with open("results/selected_features.txt") as f:
    selected_features = [line.strip() for line in f if line.strip()]

print(f"Using {len(selected_features)} selected features:")
print(selected_features)

# ---- Load data, restricted to selected features ----
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df[selected_features]
y_train = train_df["target"]
X_test = test_df[selected_features]
y_test = test_df["target"]

# ---- Load the trained XGBoost model (trained on selected features in Step 4) ----
model = joblib.load(f"{MODELS_DIR}/xgboost_selected_features.pkl")
print("\nLoaded trained XGBoost model.")

# ============================================================
# SHAP Explainer
# ============================================================
print("\nComputing SHAP values (this may take a moment)...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# ============================================================
# 1. Summary plot (beeswarm) - overall feature impact
# ============================================================
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Summary Plot - Feature Impact on Prediction")
plt.savefig(f"{OUT_DIR}/shap_summary_beeswarm.png", bbox_inches="tight", dpi=150)
plt.close()
print(f"Saved: {OUT_DIR}/shap_summary_beeswarm.png")

# ============================================================
# 2. Bar plot - mean absolute SHAP value (simple importance ranking)
# ============================================================
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Mean Absolute SHAP Value)")
plt.savefig(f"{OUT_DIR}/shap_feature_importance_bar.png", bbox_inches="tight", dpi=150)
plt.close()
print(f"Saved: {OUT_DIR}/shap_feature_importance_bar.png")

# ============================================================
# 3. Waterfall plots for a few individual patients
# One correctly predicted "sick" case, one correctly predicted "negative" case
# ============================================================
y_pred = model.predict(X_test)

# Find indices for interesting example cases
correct_sick_idx = None
correct_negative_idx = None

for i in range(len(y_test)):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    if actual == 1 and predicted == 1 and correct_sick_idx is None:
        correct_sick_idx = i
    if actual == 0 and predicted == 0 and correct_negative_idx is None:
        correct_negative_idx = i
    if correct_sick_idx is not None and correct_negative_idx is not None:
        break

explanation = shap.Explanation(
    values=shap_values,
    base_values=np.full(len(X_test), explainer.expected_value),
    data=X_test.values,
    feature_names=X_test.columns.tolist()
)

for label, idx in [("sick_example", correct_sick_idx), ("negative_example", correct_negative_idx)]:
    if idx is not None:
        plt.figure()
        shap.plots.waterfall(explanation[idx], show=False)
        plt.title(f"SHAP Waterfall - Correctly Predicted {label.replace('_', ' ').title()}")
        plt.savefig(f"{OUT_DIR}/shap_waterfall_{label}.png", bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved: {OUT_DIR}/shap_waterfall_{label}.png")

# ============================================================
# 4. Save mean absolute SHAP values as a table (for the paper)
# ============================================================
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    "Feature": selected_features,
    "Mean_Abs_SHAP": mean_abs_shap
}).sort_values(by="Mean_Abs_SHAP", ascending=False)

print("\nFeature importance ranked by mean absolute SHAP value:")
print(shap_importance_df.to_string(index=False))

shap_importance_df.to_csv(f"{OUT_DIR}/shap_importance_table.csv", index=False)
print(f"\nSaved: {OUT_DIR}/shap_importance_table.csv")

print(f"\nAll SHAP outputs saved in {OUT_DIR}/")