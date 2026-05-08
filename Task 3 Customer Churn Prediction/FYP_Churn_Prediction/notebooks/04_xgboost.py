import os
import pickle
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_recall_curve
)

# -----------------------------
# LOAD DATA
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
# TRAIN / VALID / TEST SPLIT
# -----------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
)

# Class imbalance ratio
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos

# -----------------------------
# BASE MODEL
# -----------------------------
base_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    tree_method="hist",
    scale_pos_weight=scale_pos_weight
)

# -----------------------------
# HYPERPARAMETER SEARCH
# -----------------------------
param_dist = {
    "n_estimators": [200, 300, 400, 500, 700],
    "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
    "max_depth": [3, 4, 5, 6, 8],
    "min_child_weight": [1, 3, 5, 7],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.2, 0.5, 1],
    "reg_alpha": [0, 0.01, 0.1, 1],
    "reg_lambda": [1, 1.5, 2, 3]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=25,
    scoring="roc_auc",   # change to "f1" if you want more balance, or "recall" for churn focus
    cv=cv,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)

best_model = search.best_estimator_
print("Best Parameters:", search.best_params_)
print("Best CV ROC-AUC:", search.best_score_)

# -----------------------------
# VALIDATION PROBABILITIES
# -----------------------------
val_prob = best_model.predict_proba(X_val)[:, 1]

# Find threshold that maximizes F1
precision, recall, thresholds = precision_recall_curve(y_val, val_prob)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_idx = np.argmax(f1_scores[:-1])  # thresholds is 1 shorter
best_threshold = thresholds[best_idx]

print("Best Threshold (by validation F1):", best_threshold)
print("Best Validation F1:", f1_scores[best_idx])

# -----------------------------
# TEST EVALUATION
# -----------------------------
y_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= best_threshold).astype(int)

print("\nFinal Test Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------
# SAVE FINAL MODEL + THRESHOLD
# -----------------------------
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_churn_model.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "xgboost_threshold.pkl")

with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_model, f)

with open(THRESHOLD_PATH, "wb") as f:
    pickle.dump(best_threshold, f)

print("✅ XGBoost model saved at:", MODEL_PATH)
print("✅ Threshold saved at:", THRESHOLD_PATH)