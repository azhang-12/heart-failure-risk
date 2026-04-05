import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from lifelines import KaplanMeierFitter, CoxPHFitter
from scipy.stats import percentileofscore

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Heart Failure Risk Tool",
    page_icon="🫀",
    layout="centered"
)

# ============================================================
# MODEL TRAINING
#
# This function loads the dataset and trains two models:
#
# 1. RANDOM FOREST CLASSIFIER
#    - Predicts probability of mortality from 4 clinical features
#    - Uses Platt Scaling (CalibratedClassifierCV) to convert raw
#      scores into well-calibrated probabilities
#    - Evaluated using 5-fold cross-validation for an honest AUC
#      (a single train/test split on 299 patients is too unstable)
#    - 'time' (follow-up duration) is intentionally excluded —
#      it is only known after the observation period ends, so
#      including it causes data leakage and inflates performance
#    - RF score is shown in the sidebar for academic review only,
#      not in the patient-facing results
#
# 2. COX PROPORTIONAL HAZARDS MODEL
#    - Models survival over time using 'time' as the duration axis
#    - Drives the patient-facing results:
#        * 180-day survival probability → determines risk category
#        * 30/90/180-day survival estimates
#        * Patient-specific survival curve
#    - More clinically interpretable than a raw RF probability
#      because it gives actual survival percentages at timepoints
#
# 3. SHAP EXPLAINER
#    - Explains which features drove each patient's RF score
#    - Replaces hardcoded thresholds (e.g. "if age > 70") with
#      data-driven, patient-specific explanations
#
# @st.cache_resource ensures training only runs once per session
# ============================================================

@st.cache_resource
def train_model():
    df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

    # 4 statistically significant features from univariate analysis
    # 'time' excluded from RF to prevent data leakage
    rf_features  = ['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium']
    cox_features = rf_features + ['time']

    X = df[rf_features]
    y = df['DEATH_EVENT']

    # --- Random Forest ---
    # 5-fold CV for reliable AUC across the full dataset
    rf_base = RandomForestClassifier(
        n_estimators=500,    # enough trees for stable predictions
        max_depth=6,         # prevents overfitting on small dataset
        min_samples_leaf=10, # each leaf needs at least 10 patients
        random_state=42
    )
    cv_scores  = cross_val_score(rf_base, X, y, cv=5, scoring='roc_auc')
    cv_auc     = cv_scores.mean()
    cv_auc_std = cv_scores.std()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = CalibratedClassifierCV(rf_base, method='sigmoid', cv=5)
    model.fit(X_train, y_train)

    prob_all = model.predict_proba(X)[:, 1] * 100
    df['rf_score'] = prob_all

    # --- SHAP Explainer ---
    # TreeExplainer requires the base RF, not the calibrated wrapper
    rf_base.fit(X_train, y_train)
    explainer = shap.TreeExplainer(rf_base)

    # --- Cox Proportional Hazards ---
    # 'time' = days until death or censoring (duration axis)
    # 'DEATH_EVENT' = 1 if died, 0 if censored
    cox_df = df[cox_features + ['DEATH_EVENT']].copy()
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col='time', event_col='DEATH_EVENT')

    # --- Kaplan-Meier ---
    # Population-level survival curve for reference chart only
    kmf = KaplanMeierFitter()
    kmf.fit(df['time'], event_observed=df['DEATH_EVENT'])

    return model, rf_base, explainer, cph, kmf, df, cv_auc, cv_auc_std, rf_features, cox_features


model, rf_base, explainer, cph, kmf, df, cv_auc, cv_auc_std, rf_features, cox_features = train_model()

# ============================================================
# PAGE NAVIGATION
# Sidebar radio button switches between the two pages:
#   1. Risk Assessment Tool  — patient-facing
#   2. Methodology & Assumptions — academic/professor-facing
# ============================================================

with st.sidebar:
    page = st.radio("Navigate", ["🫀 Risk Assessment Tool", "📖 Methodology & Assumptions"])

    st.divider()
    st.header("📊 Model Statistics")
    st.metric("CV AUC (5-fold)", f"{cv_auc:.3f} ± {cv_auc_std:.3f}")
    st.metric("Training Patients", "299")
    st.metric("Log-rank p-value", "< 0.000001")

    st.divider()
    st.header("📋 Hazard Ratios (Cox)")
    st.markdown("""
| Feature | HR | Effect |
|---|---|---|
| serum_creatinine | 1.390 | ↑ risk |
| age | 1.045 | ↑ risk |
| ejection_fraction | 0.956 | ↓ risk |
| serum_sodium | 0.967 | ↓ risk |
""")

    st.divider()
    st.caption("Built with Python · scikit-learn · lifelines · SHAP · Streamlit")


# ============================================================
# PAGE 1 — RISK ASSESSMENT TOOL
# ============================================================

if page == "🫀 Risk Assessment Tool":

    st.title("🫀 Heart Failure Risk Assessment Tool")
    st.markdown("### Predicting mortality risk using clinical biomarkers")
    st.markdown("""
Enter a patient's clinical values below. The tool will estimate their
**survival probability** and assign a **risk category** based on how
likely they are to survive the next 180 days.
""")
    st.divider()

    # ============================================================
    # PATIENT INPUT
    # Only 4 clinically validated features collected.
    # Selected based on statistical significance (p < 0.05)
    # in univariate analysis of the dataset.
    # ============================================================

    st.header("👤 Patient Clinical Values")
    st.markdown("Adjust the sliders to match the patient's measurements:")

    col1, col2 = st.columns(2)
    with col1:
        age               = st.slider("Age", min_value=40, max_value=95, value=60, step=1)
        ejection_fraction = st.slider("Ejection Fraction (%)", min_value=10, max_value=80, value=38, step=1)
    with col2:
        serum_creatinine = st.slider("Serum Creatinine", min_value=0.5, max_value=10.0, value=1.1, step=0.1)
        serum_sodium     = st.slider("Serum Sodium", min_value=110, max_value=150, value=137, step=1)

    # Follow-up days fixed internally — not a clinical input.
    # 130 = median follow-up in the dataset.
    FOLLOW_UP_DAYS = 130

    if st.button("🔍 Calculate Risk", type="primary", use_container_width=True):

        # --- Cox survival function ---
        patient_cox = pd.DataFrame(
            [[age, ejection_fraction, serum_creatinine, serum_sodium, FOLLOW_UP_DAYS]],
            columns=cox_features
        )
        patient_cox['DEATH_EVENT'] = 0
        survival_fn = cph.predict_survival_function(patient_cox)

        def survival_at(t):
            idx = max(0, survival_fn.index.searchsorted(t, side='right') - 1)
            return survival_fn.iloc[idx, 0] * 100

        surv_30  = survival_at(30)
        surv_90  = survival_at(90)
        surv_180 = survival_at(180)

        # Risk category based on 180-day survival
        if surv_180 < 50:
            category, color, bg = "HIGH RISK",   "🔴", "#ff4b4b"
        elif surv_180 < 75:
            category, color, bg = "MEDIUM RISK", "🟡", "#ffa500"
        else:
            category, color, bg = "LOW RISK",    "🟢", "#00c853"

        # RF score for SHAP only
        patient_rf = pd.DataFrame(
            [[age, ejection_fraction, serum_creatinine, serum_sodium]],
            columns=rf_features
        )

        st.divider()
        st.header("📋 Risk Assessment Results")

        # Risk category banner
        st.markdown(f"""
<div style='background-color:{bg}; padding:24px; border-radius:10px; text-align:center'>
  <h2 style='color:white; margin:0'>{color} {category}</h2>
  <p style='color:white; margin:10px 0 0 0; font-size:18px'>
    Estimated 180-day survival: <strong>{surv_180:.1f}%</strong>
  </p>
</div>
""", unsafe_allow_html=True)

        st.divider()

        # Survival estimates
        st.subheader("📅 Survival Estimates")
        st.caption("Estimated probability this patient is still alive at each timepoint.")
        c1, c2, c3 = st.columns(3)
        c1.metric("At 30 Days",  f"{surv_30:.1f}%")
        c2.metric("At 90 Days",  f"{surv_90:.1f}%")
        c3.metric("At 180 Days", f"{surv_180:.1f}%")

        st.divider()

        # SHAP risk drivers
        st.subheader("⚠️ Key Risk Drivers")
        st.caption("What is driving this patient's risk, ranked by importance.")

        shap_values = explainer.shap_values(patient_rf)
        if isinstance(shap_values, list):
            sv = np.array(shap_values[1]).flatten()
        else:
            sv = np.array(shap_values).flatten()

        patient_vals = [age, ejection_fraction, serum_creatinine, serum_sodium]
        pairs = sorted(zip(rf_features, sv, patient_vals), key=lambda x: abs(x[1]), reverse=True)

        for feature, shap_val, pat_val in pairs:
            direction = "↑ increasing risk" if shap_val > 0 else "↓ decreasing risk"
            icon = "🔴" if shap_val > 0.05 else ("🟢" if shap_val < -0.05 else "⚪")
            st.markdown(
                f"{icon} **{feature}** = {pat_val:.1f} &nbsp;|&nbsp; "
                f"Impact: {abs(shap_val):.3f} &nbsp;{direction}"
            )

        st.divider()

        # Survival curves
        st.subheader("📈 Survival Curves")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        kmf.plot_survival_function(ax=axes[0], ci_show=True, color='steelblue', label='All patients')
        axes[0].set_xlabel('Follow-up Days')
        axes[0].set_ylabel('Survival Probability')
        axes[0].set_title('Population Survival (Kaplan-Meier)')
        axes[0].legend()

        survival_fn.plot(ax=axes[1], color='salmon', legend=False)
        axes[1].set_xlabel('Follow-up Days')
        axes[1].set_ylabel('Survival Probability')
        axes[1].set_title("This Patient's Survival (Cox Model)")
        for t in [30, 90, 180]:
            idx = max(0, survival_fn.index.searchsorted(t, side='right') - 1)
            val = survival_fn.iloc[idx, 0]
            axes[1].axvline(x=t, color='gray', linestyle='--', alpha=0.4)
            axes[1].annotate(f'{val*100:.0f}%', xy=(t, val), fontsize=8, color='gray')

        plt.tight_layout()
        st.pyplot(fig)

    st.divider()
    st.caption("""
⚠️ This tool is for research and educational purposes only.
It should not replace clinical judgment or professional medical advice.
Risk scores are based on a dataset of 299 patients and may not
generalize to all populations.
""")


# ============================================================
# PAGE 2 — METHODOLOGY & ASSUMPTIONS
# ============================================================

elif page == "📖 Methodology & Assumptions":

    st.title("📖 Methodology & Assumptions")
    st.markdown("A transparent account of how this tool works, what decisions were made, and where the limitations lie.")
    st.divider()

    # --- Dataset ---
    st.header("1. Dataset")
    st.markdown("""
**Source:** Heart Failure Clinical Records Dataset — Chicco & Jurman (2020)

**Size:** 299 patients with heart failure, collected at Faisalabad Institute of Cardiology, Pakistan (2015)

**Outcome variable:** `DEATH_EVENT` — whether the patient died during the follow-up period (1 = died, 0 = survived)

**Follow-up period:** Varied per patient, ranging from 4 to 285 days (median ≈ 130 days)

**Important note:** Because follow-up duration varied across patients, this is not a clean "X-day mortality" prediction. 
Some patients who survived were simply discharged before 285 days — their outcome is censored, not confirmed survival.
The Cox model handles censoring correctly; the Random Forest does not distinguish between censored and confirmed survivors.
""")

    st.divider()

    # --- Feature Selection ---
    st.header("2. Feature Selection")
    st.markdown("""
The dataset contains 13 clinical features. Only 4 were used as predictors, selected based on 
**statistical significance (p < 0.05) in univariate analysis**:

| Feature | Clinical Meaning | Direction |
|---|---|---|
| `age` | Patient age in years | Older → higher risk |
| `ejection_fraction` | % of blood pumped per heartbeat | Lower → higher risk |
| `serum_creatinine` | Kidney function marker (mg/dL) | Higher → higher risk |
| `serum_sodium` | Electrolyte level (mEq/L) | Lower → higher risk |

**Excluded features:** anaemia, diabetes, high blood pressure, sex, smoking, platelets, creatinine phosphokinase — 
all failed to reach significance in univariate testing on this dataset.

**Assumption:** Univariate significance is used as a feature selection criterion. 
This is a simplification — in practice, multivariate selection or regularization (e.g. LASSO) would be more rigorous.
""")

    st.divider()

    # --- Models ---
    st.header("3. Models Used")

    st.subheader("Random Forest Classifier")
    st.markdown("""
- Trained on 4 features to predict `DEATH_EVENT`
- **Platt Scaling** (`CalibratedClassifierCV`) applied to convert raw scores to calibrated probabilities
- Hyperparameters: 500 trees, max depth 6, min 10 samples per leaf
- Used internally to power **SHAP explanations** — not shown directly to the patient
- **`time` intentionally excluded** to prevent data leakage (see below)
""")

    st.subheader("Cox Proportional Hazards Model")
    st.markdown("""
- Trained on 4 features + `time` as the duration axis
- Produces **patient-specific survival curves** — probability of surviving to any timepoint
- **Drives all patient-facing outputs:** 180-day survival, risk category, survival estimates
- **Key assumption (proportional hazards):** the model assumes that the ratio of hazard between 
  any two patients remains constant over time. This may not hold in all clinical scenarios.
""")

    st.divider()

    # --- Key Decisions ---
    st.header("4. Key Methodological Decisions")

    st.subheader("Why was 'time' excluded from the Random Forest?")
    st.markdown("""
`time` represents how long each patient was followed up — it is only known *after* the observation period ends.
Including it as a predictor would mean the model is partially "seeing the future," inflating performance 
(data leakage). It is correctly used only in the Cox model as the **duration axis**, not as a predictor.
""")

    st.subheader("Why 5-fold cross-validation instead of a single train/test split?")
    st.markdown("""
With only 299 patients, an 80/20 split leaves ~60 patients in the test set. AUC estimated on 60 patients 
has high variance — a different random seed could shift it by ±0.05. 5-fold cross-validation uses all 
patients for both training and testing (across folds), giving a much more stable and honest estimate.
""")

    st.subheader("Why does Cox drive the risk category instead of Random Forest?")
    st.markdown("""
The RF produces a raw probability (e.g. 47%) that has no intuitive time reference — 47% chance of dying 
*when*? The Cox model produces survival probabilities at specific timepoints (30, 90, 180 days), which are 
clinically interpretable. Using 180-day survival to determine risk category is more honest and meaningful 
than an arbitrary RF percentile cutoff.
""")

    st.subheader("Why SHAP for risk drivers instead of hardcoded thresholds?")
    st.markdown("""
Hardcoded thresholds (e.g. "flag if age > 70") are arbitrary and don't reflect the model's actual reasoning.
SHAP (SHapley Additive exPlanations) computes the contribution of each feature to each individual patient's 
prediction directly from the model. This means two patients with the same age can get different SHAP values 
for age depending on their other features — which is clinically realistic.
""")

    st.divider()

    # --- Assumptions & Limitations ---
    st.header("5. Assumptions & Limitations")
    st.markdown("""
| # | Assumption / Limitation | Impact |
|---|---|---|
| 1 | **Small dataset (299 patients)** — results may not generalize to other populations | High |
| 2 | **Proportional hazards assumption** — Cox assumes constant hazard ratio over time | Medium |
| 3 | **Univariate feature selection** — multivariate selection may identify different features | Medium |
| 4 | **Fixed follow-up (130 days)** used internally for Cox survival curve — individual follow-up varies | Low |
| 5 | **SHAP explains RF, not Cox** — risk drivers and survival estimates come from different models | Low |
| 6 | **Single dataset, single institution** — Faisalabad 2015, Pakistan — may not generalize globally | High |
| 7 | **Risk category thresholds (50%, 75%)** are clinically motivated but not formally validated | Medium |
""")

    st.divider()

    # --- Risk Category Thresholds ---
    st.header("6. Risk Category Thresholds")
    st.markdown("""
The risk category is determined by the **Cox-predicted 180-day survival probability**:

| Category | Threshold | Rationale |
|---|---|---|
| 🔴 High Risk | 180-day survival < 50% | Less than even odds of surviving 6 months |
| 🟡 Medium Risk | 180-day survival 50–75% | Meaningful risk, warrants close monitoring |
| 🟢 Low Risk | 180-day survival > 75% | Strong survival probability at 6 months |

**Limitation:** These thresholds were chosen based on clinical intuition, not derived from 
outcome data or validated against external cohorts. They should be interpreted as indicative, 
not diagnostic.
""")

    st.divider()

    # --- Disclaimer ---
    st.header("7. Disclaimer")
    st.markdown("""
This tool was developed as an academic research project. It is **not validated for clinical use** 
and should **not be used to make medical decisions**. 

All outputs should be interpreted by a qualified clinician in the context of the full clinical picture.
""")

    st.caption("Dataset: Chicco D, Jurman G. Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Medical Informatics and Decision Making. 2020.")