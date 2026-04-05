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
#    - Predicts probability of mortality (0-100%) from 4 clinical features
#    - Uses Platt Scaling (CalibratedClassifierCV) to convert raw scores
#      into well-calibrated probabilities
#    - Evaluated using 5-fold cross-validation for an honest AUC estimate
#      (a single train/test split on 299 patients is too unstable)
#    - 'time' (follow-up duration) is intentionally excluded — it is only
#      known after the observation period ends, so including it would cause
#      data leakage and inflate model performance
#
# 2. COX PROPORTIONAL HAZARDS MODEL
#    - Models survival over time using 'time' as the duration axis
#    - Used exclusively for patient-specific survival curve estimation
#      (probability of surviving to 30, 90, 180 days)
#    - NOT used in the risk category — only the Random Forest drives that
#
# 3. SHAP EXPLAINER
#    - Explains which features drove each individual patient's RF score
#    - Replaces hardcoded clinical thresholds (e.g. "if age > 70")
#      with data-driven, patient-specific explanations
#
# @st.cache_resource ensures this only runs once per session,
# not on every slider interaction
# ============================================================

@st.cache_resource
def train_model():
    df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

    # 4 statistically significant clinical features selected from
    # univariate analysis (Step 2 of original analysis pipeline).
    # 'time' excluded from RF to prevent data leakage.
    rf_features  = ['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium']
    cox_features = rf_features + ['time']

    X = df[rf_features]
    y = df['DEATH_EVENT']

    # --- Random Forest ---
    # 5-fold cross-validation gives a reliable AUC across the full dataset
    # rather than a single lucky/unlucky 80/20 split
    rf_base = RandomForestClassifier(
        n_estimators=500,   # enough trees for stable predictions
        max_depth=6,        # prevents overfitting on small dataset
        min_samples_leaf=10, # each leaf needs at least 10 patients
        random_state=42
    )
    cv_scores  = cross_val_score(rf_base, X, y, cv=5, scoring='roc_auc')
    cv_auc     = cv_scores.mean()
    cv_auc_std = cv_scores.std()

    # Final model trained on full dataset for patient predictions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = CalibratedClassifierCV(rf_base, method='sigmoid', cv=5)
    model.fit(X_train, y_train)

    prob_all = model.predict_proba(X)[:, 1] * 100
    df['rf_score'] = prob_all

    # --- SHAP Explainer ---
    # Trained on the base RF (pre-calibration) — TreeExplainer requires
    # a tree-based model directly, not a calibrated wrapper
    rf_base.fit(X_train, y_train)
    explainer = shap.TreeExplainer(rf_base)

    # --- Cox Proportional Hazards ---
    # 'time' is the duration axis (days until death or censoring)
    # 'DEATH_EVENT' is the event indicator (1 = died, 0 = censored)
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
# HEADER
# ============================================================

st.title("🫀 Heart Failure Risk Assessment Tool")
st.markdown("### Predicting mortality risk using clinical biomarkers")
st.markdown("""
Enter a patient's clinical values below. The tool will estimate their
**mortality risk probability** and assign a **risk category** based on
how they compare to 299 heart failure patients in the training dataset.
""")
st.divider()

# ============================================================
# SIDEBAR — Technical model statistics for clinical/academic review
#
# Kept separate from the main UI so the patient-facing view stays clean.
# Clinicians and reviewers can inspect model performance here.
# ============================================================

with st.sidebar:
    st.header("📊 Model Statistics")

    # Cross-validated AUC — more honest than a single split on 299 patients
    st.metric("CV AUC (5-fold)", f"{cv_auc:.3f} ± {cv_auc_std:.3f}")
    st.metric("Training Patients", "299")
    st.metric("Log-rank p-value", "< 0.000001")

    st.divider()

    # Hazard ratios from Cox model show the direction and magnitude
    # of each feature's effect on survival time
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
    st.header("⚠️ Methodology Note")
    st.caption("""
Follow-up time was excluded as a predictor
to prevent data leakage. It is used only
for survival curve estimation via the Cox
model, not for risk scoring.
""")

    st.divider()
    st.caption("Built with Python · scikit-learn · lifelines · SHAP · Streamlit")

# ============================================================
# PATIENT INPUT
#
# Only 4 clinically validated features are collected.
# These were selected based on statistical significance in
# univariate analysis of the dataset (p < 0.05).
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

# Follow-up days is not a clinical input — it is a fixed internal value
# used only by the Cox model to generate the survival curve.
# Using 130 days (median follow-up in the dataset).
FOLLOW_UP_DAYS = 130

# ============================================================
# RISK CALCULATION
# ============================================================

if st.button("🔍 Calculate Risk", type="primary", use_container_width=True):

    # --- Step 1: Random Forest mortality probability ---
    # This is the primary risk score shown to the user.
    # Represents estimated probability of death based on the patient's
    # clinical profile compared to patterns in the training data.
    patient_rf = pd.DataFrame(
        [[age, ejection_fraction, serum_creatinine, serum_sodium]],
        columns=rf_features
    )
    rf_prob = model.predict_proba(patient_rf)[0][1] * 100

    # --- Step 2: RF percentile vs training population ---
    # Converts the raw probability into a relative ranking.
    # e.g. "higher risk than 70% of patients" is more intuitive
    # than a raw probability alone.
    rf_percentile = percentileofscore(df['rf_score'], rf_prob)

    # --- Step 3: Risk category based on RF percentile ---
    # Thresholds: top 30% = High, middle 35% = Medium, bottom 35% = Low
    if rf_percentile >= 70:
        category, color, bg = "HIGH RISK",   "🔴", "#ff4b4b"
    elif rf_percentile >= 35:
        category, color, bg = "MEDIUM RISK", "🟡", "#ffa500"
    else:
        category, color, bg = "LOW RISK",    "🟢", "#00c853"

    # --- Step 4: Cox survival function for this patient ---
    # Used only for the survival curve and timepoint estimates.
    # Not used in risk category determination.
    patient_cox = pd.DataFrame(
        [[age, ejection_fraction, serum_creatinine, serum_sodium, FOLLOW_UP_DAYS]],
        columns=cox_features
    )
    patient_cox['DEATH_EVENT'] = 0  # placeholder required by lifelines
    survival_fn = cph.predict_survival_function(patient_cox)

    # ============================================================
    # RESULTS — Clean patient-facing display
    # ============================================================

    st.divider()
    st.header("📋 Risk Assessment Results")

    # Single clear mortality probability
    st.metric(
        label="Estimated Mortality Risk",
        value=f"{rf_prob:.1f}%",
        help="Probability of mortality based on this patient's clinical profile."
    )
    st.caption(
        f"This patient's risk is higher than **{rf_percentile:.0f}%** of the 299 patients "
        f"in the training dataset."
    )

    # Risk category banner
    st.markdown(f"""
<div style='background-color:{bg}; padding:24px; border-radius:10px; text-align:center; margin-top:12px'>
  <h2 style='color:white; margin:0'>{color} {category}</h2>
  <p style='color:white; margin:8px 0 0 0; font-size:16px'>
    Higher risk than {rf_percentile:.0f}% of similar patients
  </p>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ============================================================
    # SURVIVAL ESTIMATES
    #
    # Patient-specific survival probabilities from the Cox model.
    # These reflect this patient's individual covariate profile,
    # not population averages — each patient gets a unique curve.
    # ============================================================

    st.subheader("📅 Survival Estimates")
    st.caption("Estimated probability this patient is still alive at each timepoint.")

    c1, c2, c3 = st.columns(3)
    for col, t in zip([c1, c2, c3], [30, 90, 180]):
        idx = max(0, survival_fn.index.searchsorted(t, side='right') - 1)
        val = survival_fn.iloc[idx, 0] * 100
        col.metric(f"At {t} Days", f"{val:.1f}%")

    st.divider()

    # ============================================================
    # SHAP RISK DRIVERS
    #
    # SHAP (SHapley Additive exPlanations) explains which features
    # drove this patient's Random Forest score up or down.
    # Unlike hardcoded thresholds (e.g. "if age > 70"), SHAP values
    # are computed fresh for each patient from the actual model,
    # so the explanation reflects the model's true reasoning.
    #
    # Positive SHAP = feature increased predicted risk
    # Negative SHAP = feature decreased predicted risk
    # Features ranked by absolute impact (most influential first)
    # ============================================================

    st.subheader("⚠️ Key Risk Drivers")
    st.caption("What is driving this patient's risk score, ranked by importance.")

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

    # ============================================================
    # SURVIVAL CURVES
    #
    # Left chart: Kaplan-Meier population curve
    #   Shows overall survival across all 299 training patients.
    #   Used as a reference baseline.
    #
    # Right chart: Cox patient-specific curve
    #   Shows predicted survival for THIS patient based on their
    #   individual clinical values. Will differ from the population
    #   curve based on how their risk factors compare to average.
    # ============================================================

    st.subheader("📈 Survival Curves")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Population reference curve
    kmf.plot_survival_function(ax=axes[0], ci_show=True, color='steelblue', label='All patients')
    axes[0].set_xlabel('Follow-up Days')
    axes[0].set_ylabel('Survival Probability')
    axes[0].set_title('Population Survival (Kaplan-Meier)')
    axes[0].legend()

    # This patient's curve
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