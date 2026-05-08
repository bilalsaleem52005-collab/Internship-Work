import os
import pandas as pd
import shap
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "telco_churn.csv")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD & PREPROCESS DATA
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
df.drop(columns=["customerID"], inplace=True)

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]

scaler = StandardScaler()
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
X[num_cols] = scaler.fit_transform(X[num_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --------------------------------------------------
# TRAIN XGBOOST MODEL
# --------------------------------------------------
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=3,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)

# --------------------------------------------------
# SHAP EXPLAINABILITY
# --------------------------------------------------
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# --------------------------------------------------
# SHAP SUMMARY (BEESWARM) — SAVE
# --------------------------------------------------
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
summary_path = os.path.join(SCREENSHOT_DIR, "shap_summary.png")
plt.savefig(summary_path, dpi=300, bbox_inches="tight")
plt.close()
print("✅ Saved:", summary_path)

# --------------------------------------------------
# SHAP BAR PLOT — SAVE
# --------------------------------------------------
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
bar_path = os.path.join(SCREENSHOT_DIR, "shap_bar.png")
plt.savefig(bar_path, dpi=300, bbox_inches="tight")
plt.close()
print("✅ Saved:", bar_path)

# --------------------------------------------------
# SHAP DEPENDENCE PLOTS — SAVE
# --------------------------------------------------

# Tenure
plt.figure(figsize=(8, 6))
shap.dependence_plot("tenure", shap_values, X_test, show=False)
tenure_path = os.path.join(SCREENSHOT_DIR, "shap_dependence_tenure.png")
plt.savefig(tenure_path, dpi=300, bbox_inches="tight")
plt.close()
print("✅ Saved:", tenure_path)

# Monthly Charges
plt.figure(figsize=(8, 6))
shap.dependence_plot("MonthlyCharges", shap_values, X_test, show=False)
monthly_path = os.path.join(SCREENSHOT_DIR, "shap_dependence_monthlycharges.png")
plt.savefig(monthly_path, dpi=300, bbox_inches="tight")
plt.close()
print("✅ Saved:", monthly_path)

# Two-Year Contract
plt.figure(figsize=(8, 6))
shap.dependence_plot("Contract_Two year", shap_values, X_test, show=False)
contract_path = os.path.join(SCREENSHOT_DIR, "shap_dependence_contract_twoyear.png")
plt.savefig(contract_path, dpi=300, bbox_inches="tight")
plt.close()

print("✅ Saved:", contract_path)

# --------------------------------------------------
# TEXTUAL EXPLAINABILITY (ONE CUSTOMER)
# --------------------------------------------------
sample = X_test.iloc[0]
sample_shap = shap_values[0]

explanation = pd.Series(sample_shap, index=X_test.columns)
explanation_sorted = explanation.sort_values(ascending=False)

print("\n🔴 Top reasons customer will churn:")
print(explanation_sorted.head(5))

print("\n🟢 Top reasons customer will stay:")
print(explanation_sorted.tail(5))

