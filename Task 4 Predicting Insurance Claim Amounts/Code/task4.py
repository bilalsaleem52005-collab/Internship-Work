# =========================================================
# TASK 4 - Predicting Insurance Claim Amounts
# Linear Regression Project
# =========================================================

# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# LOAD DATASET
# =========================

# Change path if needed
df = pd.read_csv("../dataset/insurance.csv")

# =========================
# BASIC DATA EXPLORATION
# =========================

print("="*60)
print("DATASET OVERVIEW")
print("="*60)

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# =========================
# ENCODE CATEGORICAL DATA
# =========================

le = LabelEncoder()

df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])

# =========================
# CORRELATION HEATMAP
# =========================

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# =========================
# VISUALIZATIONS
# =========================

# AGE VS CHARGES
plt.figure(figsize=(8,5))
sns.scatterplot(x='age', y='charges', data=df)
plt.title("Age vs Insurance Charges")
plt.xlabel("Age")
plt.ylabel("Charges")
plt.show()

# BMI VS CHARGES
plt.figure(figsize=(8,5))
sns.scatterplot(x='bmi', y='charges', data=df)
plt.title("BMI vs Insurance Charges")
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.show()

# SMOKER VS CHARGES
plt.figure(figsize=(8,5))
sns.boxplot(x='smoker', y='charges', data=df)

plt.title("Smoking Status vs Insurance Charges")
plt.xlabel("Smoker (0 = No, 1 = Yes)")
plt.ylabel("Charges")
plt.show()

# =========================
# DEFINE FEATURES & TARGET
# =========================

X = df.drop("charges", axis=1)
y = df["charges"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================
# TRAIN LINEAR REGRESSION
# =========================

model = LinearRegression()

model.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================

y_pred = model.predict(X_test)

# =========================
# MODEL EVALUATION
# =========================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# =========================
# ACTUAL VS PREDICTED
# =========================

results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

print("\nSample Predictions:")
print(results.head(10))

# =========================
# VISUALIZE ACTUAL VS PREDICTED
# =========================

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")

plt.title("Actual vs Predicted Insurance Charges")

plt.show()

# =========================
# FEATURE IMPORTANCE
# =========================

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

coefficients = coefficients.sort_values(by='Coefficient', ascending=False)

print("\nFeature Importance:")
print(coefficients)

# =========================
# END
# =========================

print("\nTask 4 Completed Successfully!")