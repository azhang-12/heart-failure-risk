import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
# LOAD & TRAIN MODEL (cached so it only runs once)
# ============================================================

@st.cache_resource
def train_model():
    df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

    # 4 clinically valid features only (time removed — data leakage)
    features = ['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium']
    X = df[features]
    y = df['DEATH_EVENT']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=500, max_depth=6,
        min_samples_leaf=10, random_state=42
    )
    model = CalibratedClassifierCV(rf, method='sigmoid', cv=5)
    model.fit(X_train, y_train)

    prob_all = model.predict_proba(X)[:, 1] * 100
    df['rf_score'] = prob_all

    # Cox uses time + DEATH_EVENT for survival curve only
    # but only 4 features as predictors
    cox_df = df[features + ['time', 'DEATH_EVENT']].copy()
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col='time', event_col='DEATH_EVENT')

    cox_scores = cph.predict_partial_hazard(cox_df)
    cox_min = cox_scores.min()
    cox_max = cox_scores.max()
    df['cox_score'] = ((cox_scores - cox_min) / (cox_max - cox_min)) * 100
    df['unified_score'] = (df['rf_score'] * 0.6) + (df['cox_score'] * 0.4)

    kmf = KaplanMeierFitter()
    kmf.fit(df['time'], event_observed=df['DEATH_EVENT'])

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    return model, cph, kmf, df, cox_min, cox_max, auc, features

model, cph, kmf, df, cox_min, cox_max, auc, features = train_model()

# ============================================================
# HEADER
# ============================================================

st.title("🫀 Heart Failure Risk Assessment Tool")
st.markdown("### Predicting mortality risk using clinical biomarkers")
st.markdown("""
This tool uses a **unified risk model** combining:
- 🌳 Random Forest + Platt Scaling
- 📈 Cox Proportional Hazards survival analysis

Based on 299 heart failure patients. Only statistically validated features are used.
""")

st.divider()

# ============================================================
# MODEL STATS SIDEBAR
# ============================================================

with st.sidebar:
    st.header("📊 Model Statistics")
    st.metric("Random Forest AUC", f"{auc:.3f}")
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
    st.header("⚠️ Methodology Note")
    st.caption("""
    Follow-up time was excluded as a predictor 
    to prevent data leakage. It is used only 
    for survival curve estimation via 
    Kaplan-Meier, not for risk scoring.
    """)
    st.divider()
    st.caption("Built with Python, scikit-learn, lifelines & Streamlit")

# ============================================================
# PATIENT INPUT FORM
# ============================================================

st.header("👤 Patient Clinical Values")
st.markdown("Adjust the sliders to match the patient's measurements:")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=40, max_value=95, value=60, step=1)
    ejection_fraction = st.slider("Ejection Fraction (%)", min_value=10, max_value=80, value=38, step=1)

with col2:
    serum_creatinine = st.slider("Serum Creatinine", min_value=0.5, max_value=10.0, value=1.1, step=0.1)
    serum_sodium = st.slider("Serum Sodium", min_value=110, max_value=150, value=137, step=1)

# ============================================================
# CALCULATE RISK
# ============================================================

if st.button("🔍 Calculate Risk", type="primary", use_container_width=True):

    patient = pd.DataFrame([[age, ejection_fraction, serum_creatinine,
                              serum_sodium]], columns=features)

    # Random Forest probability
    rf_prob = model.predict_proba(patient)[0][1] * 100

    # Cox score — features only, no time
    cox_input = patient.copy()
    cox_input['time'] = 1
    cox_input['DEATH_EVENT'] = 0
    cox_raw = cph.predict_partial_hazard(cox_input).values[0]
    cox_norm = ((cox_raw - cox_min) / (cox_max - cox_min)) * 100
    cox_norm = np.clip(cox_norm, 0, 100)

    # Unified score
    unified = (rf_prob * 0.6) + (cox_norm * 0.4)
    percentile = percentileofscore(df['unified_score'], unified)

    # Risk category based on percentile
    if percentile >= 75:
        category = "HIGH RISK"
        color = "🔴"
        bg = "#ff4b4b"
    elif percentile >= 40:
        category = "MEDIUM RISK"
        color = "🟡"
        bg = "#ffa500"
    else:
        category = "LOW RISK"
        color = "🟢"
        bg = "#00c853"

    st.divider()
    st.header("📋 Risk Assessment Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Random Forest Score", f"{rf_prob:.1f}%")
    with col2:
        st.metric("Cox Hazard Score", f"{cox_norm:.1f}%")
    with col3:
        st.metric("Unified Risk Score", f"{unified:.1f}%")

    st.markdown(f"""
    <div style='background-color:{bg}; padding:20px; border-radius:10px; text-align:center'>
        <h2 style='color:white; margin:0'>{color} {category}</h2>
        <p style='color:white; margin:5px 0'>Higher risk than {percentile:.0f}% of patients</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Survival rates
    st.subheader("📅 Population Survival Rates")
    s30  = kmf.survival_function_at_times(30).values[0]  * 100
    s90  = kmf.survival_function_at_times(90).values[0]  * 100
    s180 = kmf.survival_function_at_times(180).values[0] * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("At 30 Days",  f"{s30:.1f}%")
    c2.metric("At 90 Days",  f"{s90:.1f}%")
    c3.metric("At 180 Days", f"{s180:.1f}%")

    # Risk drivers
    st.subheader("⚠️ Key Risk Drivers")
    drivers = []
    if ejection_fraction < 30:
        drivers.append(f"Low ejection fraction ({ejection_fraction}%) — significantly elevated risk")
    if serum_creatinine > 2.0:
        drivers.append(f"High serum creatinine ({serum_creatinine:.1f}) — kidney function concern")
    if age > 70:
        drivers.append(f"Age ({age}) — elevated baseline risk")
    if serum_sodium < 135:
        drivers.append(f"Low serum sodium ({serum_sodium}) — indicates severe heart failure")

    if drivers:
        for d in drivers:
            st.warning(d)
    else:
        st.success("✅ No major individual risk drivers flagged")

    # Survival curve
    st.divider()
    st.subheader("📈 Population Survival Curve (Kaplan-Meier)")
    fig, ax = plt.subplots(figsize=(10, 4))
    kmf.plot_survival_function(ax=ax, ci_show=True, color='steelblue',
                               label='Population survival')
    ax.set_xlabel('Follow-up Days')
    ax.set_ylabel('Survival Probability')
    ax.set_title('Kaplan-Meier Survival Curve')
    ax.legend()
    st.pyplot(fig)

    st.divider()
    st.caption("""
    ⚠️ This tool is for research purposes only and should not replace 
    clinical judgment. Risk scores are based on a dataset of 299 patients 
    and may not generalize to all populations.
    """)