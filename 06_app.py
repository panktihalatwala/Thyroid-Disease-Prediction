"""
STEP 6: Streamlit Clinical Decision-Support Prototype (multi-class,
with plain-English SHAP explanations).

IMPORTANT: The form fields below must match whatever features are
currently in results/selected_features.txt. If you rerun
04_feature_selection.py and get a different feature set, this file
needs to be updated to match.

Run with: streamlit run 06_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS_DIR = "models"

st.set_page_config(page_title="Thyroid Disease Decision-Support Prototype", layout="centered")

with open("data/target_classes.txt") as f:
    CLASS_NAMES = [line.strip() for line in f if line.strip()]

# Must match results/selected_features.txt exactly, in the same order
SELECTED_FEATURES = [
    "age", "sex", "on_thyroxine", "query_hyperthyroid", "tumor",
    "TSH_measured", "TSH", "T3_measured", "T3", "TT4", "T4U",
    "FTI_measured", "FTI", "referral_source_SVI", "referral_source_other"
]

@st.cache_resource
def load_model():
    return joblib.load(f"{MODELS_DIR}/xgboost_selected_features.pkl")

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

@st.cache_data
def load_feature_medians():
    train_df = pd.read_csv("data/processed_train.csv")
    return train_df[SELECTED_FEATURES].median()

model = load_model()
explainer = load_explainer(model)
feature_medians = load_feature_medians()

st.title("Thyroid Disease Prediction — Clinical Decision-Support Prototype")

st.warning(
    "**Academic research prototype only.** This tool does NOT provide a medical "
    "diagnosis and is not a substitute for professional medical advice."
)

st.write(
    "Enter the patient's clinical and laboratory values below. The model predicts "
    "one of four categories: negative, hyperthyroid, hypothyroid, or other, "
    "along with a plain-English explanation of why."
)

st.header("Patient Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=45)
    sex = st.selectbox("Sex", options=["Female", "Male"])
    on_thyroxine = st.selectbox("Currently on thyroxine medication?", options=["No", "Yes"])
    query_hyperthyroid = st.selectbox("Was hyperthyroidism suspected/queried by the referring doctor?", options=["No", "Yes"])
    tumor = st.selectbox("Tumor present?", options=["No", "Yes"])
    referral_source = st.selectbox(
        "Referral source",
        options=["STMW", "SVHC", "SVHD", "SVI", "WEST", "other"]
    )

with col2:
    TSH_measured = st.selectbox("Was TSH measured?", options=["Yes", "No"])
    TSH = st.number_input("TSH level", min_value=0.0, max_value=100.0, value=1.5, step=0.1)
    T3_measured = st.selectbox("Was T3 measured?", options=["Yes", "No"])
    T3 = st.number_input("T3 level", min_value=0.0, max_value=15.0, value=2.0, step=0.1)
    TT4 = st.number_input("TT4 level", min_value=0.0, max_value=300.0, value=100.0, step=1.0)
    T4U = st.number_input("T4U level", min_value=0.0, max_value=3.0, value=1.0, step=0.01)
    FTI_measured = st.selectbox("Was FTI measured?", options=["Yes", "No"])
    FTI = st.number_input("FTI level", min_value=0.0, max_value=300.0, value=100.0, step=1.0)

def yn(v): return 1 if v == "Yes" else 0
def sex_bin(v): return 1 if v == "Male" else 0

input_dict = {
    "age": age,
    "sex": sex_bin(sex),
    "on_thyroxine": yn(on_thyroxine),
    "query_hyperthyroid": yn(query_hyperthyroid),
    "tumor": yn(tumor),
    "TSH_measured": yn(TSH_measured),
    "TSH": TSH,
    "T3_measured": yn(T3_measured),
    "T3": T3,
    "TT4": TT4,
    "T4U": T4U,
    "FTI_measured": yn(FTI_measured),
    "FTI": FTI,
    "referral_source_SVI": 1 if referral_source == "SVI" else 0,
    "referral_source_other": 1 if referral_source == "other" else 0,
}
input_df = pd.DataFrame([input_dict])[SELECTED_FEATURES]

if st.button("Predict"):
    prediction_idx = model.predict(input_df)[0]
    prediction_name = CLASS_NAMES[prediction_idx]
    probabilities = model.predict_proba(input_df)[0]

    st.header("Prediction Result")
    st.subheader(f"Predicted class: **{prediction_name.upper()}**")

    prob_df = pd.DataFrame({"Class": CLASS_NAMES, "Probability": probabilities}).sort_values(
        by="Probability", ascending=False
    )
    st.bar_chart(prob_df.set_index("Class"))
    st.dataframe(prob_df.style.format({"Probability": "{:.1%}"}), hide_index=True)

    st.header("Why did the model predict this?")

    explanation = explainer(input_df)
    shap_row = explanation.values[0, :, prediction_idx]

    impact_order = np.argsort(-np.abs(shap_row))[:3]
    phrases = []
    for i in impact_order:
        feat = SELECTED_FEATURES[i]
        impact = shap_row[i]
        value = input_df.iloc[0][feat]
        median = feature_medians[feat]
        direction = "pushed toward" if impact > 0 else "pushed away from"
        level = "higher than typical" if value > median else "lower than typical" if value < median else "close to typical"
        phrases.append(f"**{feat}** = {value:.2f} ({level}) {direction} this prediction")

    st.write(
        f"This case was predicted as **{prediction_name}** mainly because of: "
        + "; ".join(phrases) + "."
    )

    single_explanation = shap.Explanation(
        values=shap_row,
        base_values=explanation.base_values[0, prediction_idx],
        data=input_df.iloc[0].values,
        feature_names=SELECTED_FEATURES
    )
    fig = plt.figure()
    shap.plots.waterfall(single_explanation, show=False)
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "This explanation reflects the model's reasoning based on patterns in the "
        "training data, not a medical diagnosis. Always consult a qualified "
        "healthcare professional for medical decisions."
    )