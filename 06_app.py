"""
STEP 6: Streamlit Clinical Decision-Support Prototype

Loads the trained XGBoost model (on 15 selected features), lets a user
enter clinical/lab values, and shows:
  - Predicted class (sick / negative)
  - Prediction confidence
  - SHAP explanation of which features drove this specific prediction

IMPORTANT: This is an academic research prototype. It is NOT a medical
diagnosis tool and must not be presented or used as one.

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

# ---- Load model and scaler ----
@st.cache_resource
def load_model():
    model = joblib.load(f"{MODELS_DIR}/xgboost_selected_features.pkl")
    return model

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

model = load_model()
explainer = load_explainer(model)

SELECTED_FEATURES = [
    "age", "sex", "on_thyroxine", "pregnant", "TSH_measured", "TSH",
    "T3_measured", "T3", "TT4_measured", "TT4", "T4U", "FTI",
    "referral_source_SVHC", "referral_source_SVI", "referral_source_other"
]

# ============================================================
# HEADER + DISCLAIMER
# ============================================================
st.title("Thyroid Disease Prediction — Clinical Decision-Support Prototype")

st.warning(
    "**Academic research prototype only.** This tool does NOT provide a medical "
    "diagnosis and is not a substitute for professional medical advice. "
    "It was built as part of a college research methodology project."
)

st.write(
    "Enter the patient's clinical and laboratory values below. The model will "
    "predict whether the case is likely 'negative' (no thyroid condition flagged) "
    "or 'sick' (a thyroid-related condition flagged), along with an explanation "
    "of which factors contributed most to that prediction."
)

# ============================================================
# INPUT FORM
# ============================================================
st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=45)
    sex = st.selectbox("Sex", options=["Female", "Male"])
    on_thyroxine = st.selectbox("Currently on thyroxine medication?", options=["No", "Yes"])
    pregnant = st.selectbox("Pregnant?", options=["No", "Yes"])
    referral_source = st.selectbox(
        "Referral source",
        options=["Other", "SVHC", "SVI", "SVHD", "WEST"]
    )

with col2:
    TSH_measured = st.selectbox("Was TSH measured?", options=["Yes", "No"])
    TSH = st.number_input("TSH level", min_value=0.0, max_value=100.0, value=1.5, step=0.1)
    T3_measured = st.selectbox("Was T3 measured?", options=["Yes", "No"])
    T3 = st.number_input("T3 level", min_value=0.0, max_value=15.0, value=2.0, step=0.1)
    TT4_measured = st.selectbox("Was TT4 measured?", options=["Yes", "No"])
    TT4 = st.number_input("TT4 level", min_value=0.0, max_value=300.0, value=100.0, step=1.0)
    T4U = st.number_input("T4U level", min_value=0.0, max_value=3.0, value=1.0, step=0.01)
    FTI = st.number_input("FTI level", min_value=0.0, max_value=300.0, value=100.0, step=1.0)

# ============================================================
# BUILD INPUT ROW MATCHING TRAINING FEATURE FORMAT
# ============================================================
def yn_to_binary(value):
    return 1 if value == "Yes" else 0

def sex_to_binary(value):
    return 1 if value == "Male" else 0

input_dict = {
    "age": age,
    "sex": sex_to_binary(sex),
    "on_thyroxine": yn_to_binary(on_thyroxine),
    "pregnant": yn_to_binary(pregnant),
    "TSH_measured": yn_to_binary(TSH_measured),
    "TSH": TSH,
    "T3_measured": yn_to_binary(T3_measured),
    "T3": T3,
    "TT4_measured": yn_to_binary(TT4_measured),
    "TT4": TT4,
    "T4U": T4U,
    "FTI": FTI,
    "referral_source_SVHC": 1 if referral_source == "SVHC" else 0,
    "referral_source_SVI": 1 if referral_source == "SVI" else 0,
    "referral_source_other": 1 if referral_source == "Other" else 0,
}

input_df = pd.DataFrame([input_dict])[SELECTED_FEATURES]

# ============================================================
# PREDICTION
# ============================================================
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.header("Prediction Result")

    if prediction == 1:
        st.error(f"**Predicted class: SICK** (thyroid condition flagged)")
    else:
        st.success(f"**Predicted class: NEGATIVE** (no thyroid condition flagged)")

    st.write(f"Model confidence — Negative: {probability[0]:.1%} | Sick: {probability[1]:.1%}")

    # ---- SHAP explanation for this specific prediction ----
    st.header("Why did the model predict this?")
    st.write(
        "The chart below shows which factors pushed the prediction toward "
        "'sick' (red, right) or 'negative' (blue, left), and by how much."
    )

    shap_values_single = explainer.shap_values(input_df)

    explanation = shap.Explanation(
        values=shap_values_single[0],
        base_values=explainer.expected_value,
        data=input_df.iloc[0].values,
        feature_names=input_df.columns.tolist()
    )

    fig = plt.figure()
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "This explanation reflects the model's reasoning, not a medical diagnosis. "
        "Always consult a qualified healthcare professional for medical decisions."
    )