import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# FEATURES & TARGET
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Scale numeric features
scaler = StandardScaler()
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
X[num_cols] = scaler.fit_transform(X[num_cols])

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# TUNED RANDOM FOREST as earlier it was worse than LR
# -----------------------------

rf_model = RandomForestClassifier(
    n_estimators=500,          # more trees = better learning
    max_depth=10,              # prevent overfitting
    min_samples_split=10,      # force meaningful splits
    min_samples_leaf=5,        # smoother predictions
    class_weight={0: 1, 1: 3}, # focus more on churn
    random_state=42
)

rf_model.fit(X_train, y_train)


# Predicitions 

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

# Evauation 
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

#Although Random Forest is a more complex model, its default
#  configuration underperformed compared to Logistic Regression,
#  highlighting the importance of hyperparameter tuning. 
# I compared models honestly.
#  So i tuned it