import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# STEP 1 — EXPLORATORY DATA ANALYSIS
# ============================================================

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Total patients: {len(df)}")
print(f"Deaths: {df['DEATH_EVENT'].sum()} ({df['DEATH_EVENT'].mean()*100:.1f}%)")
print(f"Survived: {(df['DEATH_EVENT']==0).sum()} ({(df['DEATH_EVENT']==0).mean()*100:.1f}%)")
print(f"\nFeatures: {list(df.columns)}")
print("\nBasic Statistics:")
print(df.describe().round(2))

# Check for missing values
print("\n" + "=" * 60)
print("MISSING VALUES CHECK")
print("=" * 60)
print(df.isnull().sum())

# Separate continuous and binary features
continuous = ['age', 'creatinine_phosphokinase', 'ejection_fraction',
              'platelets', 'serum_creatinine', 'serum_sodium', 'time']
binary = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

# ============================================================
# STEP 2 — STATISTICAL FEATURE TESTING
# ============================================================

print("\n" + "=" * 60)
print("STEP 2 — FEATURE IMPORTANCE (STATISTICAL TESTS)")
print("=" * 60)

# Split into died vs survived groups
died = df[df['DEATH_EVENT'] == 1]
survived = df[df['DEATH_EVENT'] == 0]

print(f"\nDied group: {len(died)} patients")
print(f"Survived group: {len(survived)} patients")

# --- Continuous variables: Mann-Whitney U test ---
print("\n" + "-" * 60)
print("CONTINUOUS VARIABLES (Mann-Whitney U Test)")
print("-" * 60)
print(f"{'Feature':<30} {'Died Mean':>12} {'Survived Mean':>15} {'P-Value':>10} {'Significant':>12}")
print("-" * 60)

continuous_results = []
for feature in continuous:
    died_vals = died[feature]
    survived_vals = survived[feature]
    stat, p_value = stats.mannwhitneyu(died_vals, survived_vals)
    significant = "YES ✓" if p_value < 0.05 else "NO"
    continuous_results.append((feature, p_value, significant))
    print(f"{feature:<30} {died_vals.mean():>12.2f} {survived_vals.mean():>15.2f} {p_value:>10.4f} {significant:>12}")

# --- Binary variables: Chi-square test ---
print("\n" + "-" * 60)
print("BINARY VARIABLES (Chi-Square Test)")
print("-" * 60)
print(f"{'Feature':<30} {'Died %':>10} {'Survived %':>12} {'P-Value':>10} {'Significant':>12}")
print("-" * 60)

binary_results = []
for feature in binary:
    contingency = pd.crosstab(df[feature], df['DEATH_EVENT'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    died_pct = died[feature].mean() * 100
    survived_pct = survived[feature].mean() * 100
    significant = "YES ✓" if p_value < 0.05 else "NO"
    binary_results.append((feature, p_value, significant))
    print(f"{feature:<30} {died_pct:>9.1f}% {survived_pct:>11.1f}% {p_value:>10.4f} {significant:>12}")

# --- Summary: Significant features ---
print("\n" + "=" * 60)
print("SUMMARY — STATISTICALLY SIGNIFICANT FEATURES")
print("=" * 60)
significant_features = []
for feature, p_value, significant in continuous_results + binary_results:
    if significant == "YES ✓":
        significant_features.append(feature)
        print(f"  ✓ {feature} (p={p_value:.4f})")

print(f"\nFeatures to use in model: {significant_features}")

# ============================================================
# VISUALIZATION — Distribution plots for continuous variables
# ============================================================

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Feature Distributions: Died vs Survived', fontsize=14)
axes = axes.flatten()

for i, feature in enumerate(continuous):
    axes[i].hist(survived[feature], alpha=0.6, label='Survived', color='steelblue', bins=20)
    axes[i].hist(died[feature], alpha=0.6, label='Died', color='salmon', bins=20)
    axes[i].set_title(feature)
    axes[i].legend()

plt.tight_layout()
plt.savefig('step1_distributions.png')
plt.show()
print("\nDistribution chart saved as step1_distributions.png!")
