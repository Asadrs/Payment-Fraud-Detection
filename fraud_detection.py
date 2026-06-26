from sklearn.metrics import roc_curve
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/creditcard.csv")

print("Dataset Shape:", df.shape)

# Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Class", data=df)

plt.title("Class Distribution")
plt.xticks([0,1], ["Legitimate", "Fraud"])
plt.xlabel("Transaction Type")
plt.ylabel("Count")

plt.savefig("outputs/class_distribution.png")
plt.show()

# Transaction Amount Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Amount"], bins=50)

plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")

plt.savefig("outputs/amount_distribution.png")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(14,10))

corr = df.corr()

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap")

plt.savefig("outputs/correlation_heatmap.png")
plt.close()

# -----------------------------
# Data Preprocessing
# -----------------------------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Scale Amount and Time
scaler = StandardScaler()

X["Amount"] = scaler.fit_transform(X[["Amount"]])
X["Time"] = scaler.fit_transform(X[["Time"]])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Set Shape:")
print(X_train.shape)

print("\nTesting Set Shape:")
print(X_test.shape)

print("\nTraining Class Distribution:")
print(y_train.value_counts())

print("\nTesting Class Distribution:")
print(y_test.value_counts())

# -----------------------------
# Handle Imbalanced Data using SMOTE
# -----------------------------

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nBefore SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

# -----------------------------
# Logistic Regression Model
# -----------------------------

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train_smote, y_train_smote)

print("\nModel Training Completed.")

# Predictions

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# -----------------------------
# Model Evaluation
# -----------------------------

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nROC AUC Score:")
print(roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# Confusion Matrix Heatmap
# -----------------------------

plt.figure(figsize=(6,5))

sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix.png")
plt.close()

# -----------------------------
# ROC Curve
# -----------------------------

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7,5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc_score(y_test, y_prob):.4f}"
)

plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("outputs/roc_curve.png")
plt.close()

print("\nGraphs Saved Successfully.")

# -----------------------------
# Random Forest Model
# -----------------------------

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_smote, y_train_smote)

rf_pred = rf_model.predict(X_test)

rf_prob = rf_model.predict_proba(X_test)[:, 1]

print("\n========== RANDOM FOREST ==========")

print("\nAccuracy:")
print(accuracy_score(y_test, rf_pred))

print("\nROC AUC:")
print(roc_auc_score(y_test, rf_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

# -----------------------------
# Save Model
# -----------------------------

import joblib

joblib.dump(rf_model, "models/fraud_model.pkl")

print("\nModel Saved Successfully.")