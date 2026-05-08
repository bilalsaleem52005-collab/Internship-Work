
import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------
# LOAD DATASET
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "telco_churn.csv")

df = pd.read_csv(DATA_PATH)

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# Encode target
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Drop ID
df.drop(columns=["customerID"], inplace=True)

# One-hot encode
df = pd.get_dummies(df, drop_first=True)

# -----------------------------
# FEATURES & TARGET
# -----------------------------

X = df.drop("Churn", axis=1)
y = df["Churn"]

# -----------------------------
# FEATURE SCALING
# -----------------------------

scaler = StandardScaler()
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

X[num_cols] = scaler.fit_transform(X[num_cols])

# -----------------------------
# TRAIN-TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# TRAIN LOGISTIC REGRESSION
# -----------------------------

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# PREDICTIONS
# -----------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -----------------------------
# EVALUATION
# -----------------------------

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
