from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).parent / "selected_heart_disease_pipeline.joblib"

st.set_page_config(
    page_title="Heart Disease AI Learning Lab",
    page_icon="🫀",
    layout="wide",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


st.title("Heart Disease AI Learning Lab")
st.error(
    "For education and research demonstration only. This application must not be "
    "used for diagnosis, treatment selection, screening, or patient management."
)
st.caption(
    "Model: Logistic Regression selected by five-fold cross-validation AUROC | "
    "Dataset: UCI Heart Disease (Cleveland subset)"
)

with st.expander("About this educational experiment", expanded=False):
    st.markdown(
        """
        The model was trained on 303 historical records from the UCI Cleveland Heart
        Disease dataset. The original diagnosis label (0–4) was converted to a binary
        educational target: 0 = absence and >0 = presence. The dataset is small,
        historical, and not representative of present-day populations. Predictions
        illustrate model behaviour; they are not medical risk estimates.

        **Verified experiment results**

        - Five-fold CV accuracy: **0.8552 ± 0.0238**
        - Five-fold CV AUROC: **0.9025 ± 0.0148**
        - Holdout accuracy: **0.8689**
        - Holdout sensitivity: **0.9286**
        - Holdout specificity: **0.8182**
        - Holdout AUROC: **0.9665**
        """
    )

st.subheader("Step 1 — Record your judgment before viewing the AI result")
judgment_col, confidence_col = st.columns(2)
with judgment_col:
    learner_judgment = st.radio(
        "Your initial classification",
        ["Absence", "Presence"],
        horizontal=True,
    )
with confidence_col:
    learner_confidence = st.slider("Confidence in your judgment (%)", 0, 100, 50)

learner_rationale = st.text_area(
    "Brief rationale",
    placeholder="Which features influenced your judgment?",
    max_chars=500,
)

st.subheader("Step 2 — Enter a simulated case")
st.caption("Use only simulated values. Do not enter identifiable or real patient information.")

with st.form("case_form"):
    left, middle, right = st.columns(3)

    with left:
        age = st.number_input("Age (years)", 20, 90, 54)
        sex = st.selectbox("Recorded sex code", [(0, "0 — Female"), (1, "1 — Male")], format_func=lambda x: x[1])
        cp = st.selectbox(
            "Chest-pain type",
            [(1, "1 — Typical angina"), (2, "2 — Atypical angina"),
             (3, "3 — Non-anginal pain"), (4, "4 — Asymptomatic")],
            format_func=lambda x: x[1],
        )
        trestbps = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 130)
        chol = st.number_input("Serum cholesterol (mg/dL)", 100, 600, 245)

    with middle:
        fbs = st.selectbox("Fasting blood sugar >120 mg/dL", [(0, "0 — No"), (1, "1 — Yes")], format_func=lambda x: x[1])
        restecg = st.selectbox(
            "Resting ECG result",
            [(0, "0 — Normal"), (1, "1 — ST-T abnormality"), (2, "2 — LV hypertrophy")],
            format_func=lambda x: x[1],
        )
        thalach = st.number_input("Maximum heart rate achieved", 60, 220, 150)
        exang = st.selectbox("Exercise-induced angina", [(0, "0 — No"), (1, "1 — Yes")], format_func=lambda x: x[1])
        oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)

    with right:
        slope = st.selectbox(
            "Slope of peak exercise ST segment",
            [(1, "1 — Upsloping"), (2, "2 — Flat"), (3, "3 — Downsloping")],
            format_func=lambda x: x[1],
        )
        ca = st.selectbox("Number of major vessels", [0, 1, 2, 3])
        thal = st.selectbox(
            "Thal category",
            [(3, "3 — Normal"), (6, "6 — Fixed defect"), (7, "7 — Reversible defect")],
            format_func=lambda x: x[1],
        )
        acknowledge = st.checkbox(
            "I confirm that this is a simulated educational case and not a real patient."
        )

    submitted = st.form_submit_button("Compare my judgment with the AI model", type="primary")


if submitted:
    if not acknowledge:
        st.warning("Confirm that the values represent a simulated educational case.")
        st.stop()

    model = load_model()
    case = pd.DataFrame(
        [{
            "age": age,
            "sex": sex[0],
            "cp": cp[0],
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs[0],
            "restecg": restecg[0],
            "thalach": thalach,
            "exang": exang[0],
            "oldpeak": oldpeak,
            "slope": slope[0],
            "ca": ca,
            "thal": thal[0],
        }]
    )

    probability = float(model.predict_proba(case)[0, 1])
    predicted_class = int(model.predict(case)[0])
    predicted_label = "Presence" if predicted_class == 1 else "Absence"
    learner_class = 1 if learner_judgment == "Presence" else 0

    st.subheader("Step 3 — Compare and critique")
    m1, m2, m3 = st.columns(3)
    m1.metric("Your judgment", learner_judgment)
    m2.metric("AI classification", predicted_label)
    m3.metric("Model output", f"{probability:.1%}", help="Model probability for the positive class; not a clinical risk estimate.")

    if learner_class == predicted_class:
        st.success("Your classification agrees with the model. Agreement does not establish correctness.")
    else:
        st.warning("Your classification differs from the model. Review the inputs and decide whether disagreement is justified.")

    if probability < 0.40:
        band = "lower model-output band"
    elif probability > 0.60:
        band = "higher model-output band"
    else:
        band = "uncertain model-output band"
    st.info(f"The case falls in the **{band}**. The threshold used for classification is 0.50.")

    try:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        transformed = preprocessor.transform(case)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        feature_names = preprocessor.get_feature_names_out()
        contributions = transformed[0] * classifier.coef_[0]
        contribution_df = pd.DataFrame(
            {"Transformed feature": feature_names, "Contribution": contributions}
        )
        contribution_df["Magnitude"] = contribution_df["Contribution"].abs()
        contribution_df = contribution_df.nlargest(8, "Magnitude").drop(columns="Magnitude")
        contribution_df = contribution_df.sort_values("Contribution")

        st.markdown("#### Largest model contributions for this case")
        st.bar_chart(contribution_df.set_index("Transformed feature")["Contribution"])
        st.caption(
            "Positive values push the logistic-regression output toward the positive class; "
            "negative values push it toward the negative class. These are model associations, not causal effects."
        )
    except Exception:
        st.caption("Detailed contribution display is unavailable for this model version.")

    st.markdown("#### Reflection questions")
    st.markdown(
        f"""
        1. You reported **{learner_confidence}% confidence**. Was that confidence justified?
        2. Which two variables most influenced the model output?
        3. Could missing context make either your judgment or the model output unreliable?
        4. Why should this output not be interpreted as a diagnosis?
        """
    )
    if learner_rationale.strip():
        st.caption(f"Your original rationale: {learner_rationale}")

st.divider()
st.markdown(
    "**Dataset citation:** Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. "
    "(1989). *Heart Disease*. UCI Machine Learning Repository. "
    "https://doi.org/10.24432/C52P4X"
)
st.caption("Authors: Arivoli Sundaramurthy and Chitra Vaithiyalingam, PSG Institute of Technology and Applied Research, Coimbatore, India.")

