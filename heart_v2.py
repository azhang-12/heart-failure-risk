import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Load the dataset
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Prepare features and target
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# Scale the data (important for probability curves)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Logistic Regression (best for probability output)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Get probability scores (0-100%)
probabilities = model.predict_proba(X_test)[:, 1] * 100

# Print results
print("Patient Risk Probabilities (Death Event %):")
for i, prob in enumerate(probabilities):
    risk = "HIGH" if prob >= 60 else "MEDIUM" if prob >= 30 else "LOW"
    print(f"Patient {i+1}: {prob:.1f}% risk [{risk}]")

# AUC Score (model quality, closer to 1.0 is better)
auc = roc_auc_score(y_test, probabilities)
print(f"\nModel AUC Score: {auc:.2f}")

# Risk Factor Importance Chart
feature_names = X.columns
importance = abs(model.coef_[0])

plt.figure(figsize=(10, 6))
sorted_idx = importance.argsort()
plt.barh(feature_names[sorted_idx], importance[sorted_idx], color='steelblue')
plt.xlabel('Impact on Death Risk')
plt.title('Risk Factor Importance Chart')
plt.tight_layout()
plt.savefig('risk_factors.png')
plt.show()
print("\nChart saved as risk_factors.png!")
