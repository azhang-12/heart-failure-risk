import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from scipy.stats import percentileofscore

# ============================================================
# LOAD & PREPARE DATA
# ============================================================

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Only statistically significant features (from Step 2)
features = ['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time']
X = df[features]
y = df['DEATH_EVENT']

# ============================================================
# STEP 3 — RANDOM FOREST + PLATT SCALING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=6,
    min_samples_leaf=10,
    random_state=42
)

model = CalibratedClassifierCV(rf, method='sigmoid', cv=5)
model.fit(X_train, y_train)

prob_all = model.predict_proba(X)[:, 1] * 100
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
df['rf_score'] = prob_all

# ============================================================
# STEP 4 — KAPLAN-MEIER
# ============================================================

kmf = KaplanMeierFitter()
kmf.fit(df['time'], event_observed=df['DEATH_EVENT'])

# ============================================================
# STEP 5 — COX PROPORTIONAL HAZARDS
# ============================================================

cox_df = df[features + ['DEATH_EVENT']].copy()
cph = CoxPHFitter()
cph.fit(cox_df, duration_col='time', event_col='DEATH_EVENT')

cox_scores = cph.predict_partial_hazard(cox_df)

# Normalize Cox score to 0-100%
cox_min = cox_scores.min()
cox_max = cox_scores.max()
df['cox_score'] = ((cox_scores - cox_min) / (cox_max - cox_min)) * 100

# ============================================================
# STEP 6 — UNIFIED RISK SCORE (RF 60% + Cox 40%)
# ============================================================

df['unified_score'] = (df['rf_score'] * 0.6) + (df['cox_score'] * 0.4)

# Validate unified score
median_unified = df['unified_score'].median()
df['risk_group'] = df['unified_score'].apply(
    lambda x: 'High Risk' if x >= median_unified else 'Low Risk'
)

high_risk = df[df['risk_group'] == 'High Risk']
low_risk  = df[df['risk_group'] == 'Low Risk']

results = logrank_test(
    high_risk['time'], low_risk['time'],
    event_observed_A=high_risk['DEATH_EVENT'],
    event_observed_B=low_risk['DEATH_EVENT']
)

# ============================================================
# PRINT MODEL SUMMARY
# ============================================================

print("=" * 60)
print("   HEART FAILURE RISK MODEL — SUMMARY")
print("=" * 60)
print(f"  Total patients trained on : 299")
print(f"  Significant features used : {features}")
print(f"  Random Forest AUC         : {auc:.3f}")
print(f"  Cox Concordance           : {cph.concordance_index_:.3f}")
print(f"  Unified Score AUC         : {roc_auc_score(y, (df['unified_score']/100)):.3f}")
print(f"  Log-rank p-value          : {results.p_value:.6f}")
print(f"\n  High Risk death rate      : {high_risk['DEATH_EVENT'].mean()*100:.1f}%")
print(f"  Low Risk death rate       : {low_risk['DEATH_EVENT'].mean()*100:.1f}%")

print("\n  Hazard Ratios (Cox):")
hr = np.exp(cph.params_)
for feature, ratio in hr.items():
    direction = "↑ increases risk" if ratio > 1 else "↓ decreases risk"
    print(f"    {feature:<25} HR={ratio:.3f}  {direction}")

# ============================================================
# SURVIVAL CURVES
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Heart Failure Risk Model', fontsize=14)

# Overall KM curve
kmf.plot_survival_function(ax=axes[0], ci_show=True, color='steelblue')
axes[0].set_title('Overall Survival Curve (Kaplan-Meier)')
axes[0].set_xlabel('Follow-up Days')
axes[0].set_ylabel('Survival Probability')
axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

# High vs Low risk KM curves
kmf_high = KaplanMeierFitter()
kmf_low  = KaplanMeierFitter()
kmf_high.fit(high_risk['time'], event_observed=high_risk['DEATH_EVENT'], label='High Risk')
kmf_low.fit(low_risk['time'],  event_observed=low_risk['DEATH_EVENT'],  label='Low Risk')
kmf_high.plot_survival_function(ax=axes[1], ci_show=True, color='salmon')
kmf_low.plot_survival_function(ax=axes[1],  ci_show=True, color='steelblue')
axes[1].set_title('Survival: High Risk vs Low Risk (Unified Score)')
axes[1].set_xlabel('Follow-up Days')
axes[1].set_ylabel('Survival Probability')

plt.tight_layout()
plt.savefig('survival_curves.png')
plt.show()
print("\n  Survival curves saved as survival_curves.png")

# ============================================================
# INTERACTIVE PATIENT INPUT TOOL
# ============================================================

def get_float_input(prompt, min_val, max_val):
    while True:
        try:
            val = float(input(prompt))
            if min_val <= val <= max_val:
                return val
            print(f"  Please enter a value between {min_val} and {max_val}")
        except ValueError:
            print("  Please enter a valid number")

def assess_patient():
    print("\n" + "=" * 60)
    print("   HEART FAILURE RISK ASSESSMENT TOOL")
    print("=" * 60)
    print("  Please enter the patient's clinical values:")
    print("-" * 60)

    age               = get_float_input("  Age (40-95): ", 40, 95)
    ejection_fraction = get_float_input("  Ejection Fraction % (10-80): ", 10, 80)
    serum_creatinine  = get_float_input("  Serum Creatinine (0.5-10.0): ", 0.5, 10.0)
    serum_sodium      = get_float_input("  Serum Sodium (110-150): ", 110, 150)
    time              = get_float_input("  Follow-up Days (1-285): ", 1, 285)

    # Build patient dataframe
    patient = pd.DataFrame([[age, ejection_fraction, serum_creatinine,
                              serum_sodium, time]], columns=features)

    # RF probability
    rf_prob = model.predict_proba(patient)[0][1] * 100

    # Cox hazard score — normalized using training data range
    cox_raw = cph.predict_partial_hazard(
        pd.concat([patient.assign(DEATH_EVENT=0)], ignore_index=True)
    ).values[0]
    cox_norm = ((cox_raw - cox_min) / (cox_max - cox_min)) * 100
    cox_norm = np.clip(cox_norm, 0, 100)

    # Unified score
    unified = (rf_prob * 0.6) + (cox_norm * 0.4)

    # Percentile vs training data
    percentile = percentileofscore(df['unified_score'], unified)

# Risk category based on percentile (more honest than raw score)
    if percentile >= 75:
        category = "HIGH RISK 🔴"
    elif percentile >= 40:
        category = "MEDIUM RISK 🟡"
    else:
        category = "LOW RISK 🟢"

    # Survival probability at key timepoints
    survival_30  = kmf.survival_function_at_times(30).values[0]  * 100
    survival_90  = kmf.survival_function_at_times(90).values[0]  * 100
    survival_180 = kmf.survival_function_at_times(180).values[0] * 100

    print("\n" + "=" * 60)
    print("   RISK ASSESSMENT RESULTS")
    print("=" * 60)
    print(f"  Patient Profile:")
    print(f"    Age                : {age:.0f}")
    print(f"    Ejection Fraction  : {ejection_fraction:.0f}%")
    print(f"    Serum Creatinine   : {serum_creatinine:.2f}")
    print(f"    Serum Sodium       : {serum_sodium:.0f}")
    print(f"    Follow-up Days     : {time:.0f}")
    print("-" * 60)
    print(f"  Random Forest Score : {rf_prob:.1f}%")
    print(f"  Cox Hazard Score    : {cox_norm:.1f}%")
    print(f"  ► Unified Risk Score: {unified:.1f}%")
    print(f"  ► Risk Category     : {category}")
    print(f"  ► Percentile        : Higher risk than {percentile:.0f}% of patients")
    print("-" * 60)
    print(f"  Population Survival Rates (from dataset):")
    print(f"    At 30 days  : {survival_30:.1f}%")
    print(f"    At 90 days  : {survival_90:.1f}%")
    print(f"    At 180 days : {survival_180:.1f}%")
    print("=" * 60)

    # Key risk drivers for this patient
    print("\n  Key Risk Drivers for This Patient:")
    if ejection_fraction < 30:
        print(f"  ⚠️  Low ejection fraction ({ejection_fraction:.0f}%) — significantly elevated risk")
    if serum_creatinine > 2.0:
        print(f"  ⚠️  High serum creatinine ({serum_creatinine:.2f}) — kidney function concern")
    if age > 70:
        print(f"  ⚠️  Age ({age:.0f}) — elevated baseline risk")
    if serum_sodium < 135:
        print(f"  ⚠️  Low serum sodium ({serum_sodium:.0f}) — indicates severe heart failure")
    if time < 50:
        print(f"  ⚠️  Short follow-up ({time:.0f} days) — early period is highest risk window")

# ============================================================
# RUN THE TOOL
# ============================================================

while True:
    assess_patient()
    again = input("\n  Assess another patient? (yes/no): ").strip().lower()
    if again != 'yes':
        print("\n  Thank you for using the Heart Failure Risk Tool.")
        break