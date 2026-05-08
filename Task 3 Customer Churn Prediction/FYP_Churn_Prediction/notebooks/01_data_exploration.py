import os
import pandas as pd

# ✅ Always build paths relative to THIS script file (works from anywhere)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))              # .../notebooks
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))        # .../FYP_Churn_Prediction
DATA_PATH = os.path.join(PROJECT_DIR, "data", "telco_churn.csv")     # .../data/telco_churn.csv

print("Script directory:", SCRIPT_DIR)
print("Project directory:", PROJECT_DIR)
print("Dataset path:", DATA_PATH)

# ✅ Check dataset exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Make sure telco_churn.csv is inside the 'data' folder."
    )

# ✅ Load dataset
df = pd.read_csv(DATA_PATH)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset info:")
df.info()

print("\nDataset shape (rows, columns):")
print(df.shape)

print("\nChurn distribution:")
print(df["Churn"].value_counts())

# -----------------------------
# FIX TotalCharges COLUMN
# -----------------------------

# Convert TotalCharges to numeric (invalid values → NaN)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Check how many values became NaN
print("\nMissing values after conversion:")
print(df["TotalCharges"].isnull().sum())

# -----------------------------
# REMOVE ROWS WITH MISSING TotalCharges
# -----------------------------

before_rows = df.shape[0]

df.dropna(subset=["TotalCharges"], inplace=True)

after_rows = df.shape[0]

print(f"\nRows before cleaning: {before_rows}")
print(f"Rows after cleaning: {after_rows}")

print("\nUpdated data types:")
print(df.dtypes)

# -----------------------------
# ENCODE TARGET VARIABLE
# -----------------------------

df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

print("\nChurn value counts after encoding:")
print(df["Churn"].value_counts())

# -----------------------------
# DROP CUSTOMER ID
# -----------------------------

df.drop(columns=["customerID"], inplace=True)

# -----------------------------
# ONE-HOT ENCODE CATEGORICAL FEATURES
# -----------------------------

df = pd.get_dummies(df, drop_first=True)

print("\nDataset shape after encoding:")
print(df.shape)

print("\nAny missing values left?")
print(df.isnull().sum().sum())

# -----------------------------
# SPLIT FEATURES AND TARGET
# -----------------------------

X = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nFeature matrix shape (X):", X.shape)
print("Target vector shape (y):", y.shape)

from sklearn.preprocessing import StandardScaler

# -----------------------------
# FEATURE SCALING
# -----------------------------

scaler = StandardScaler()

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

X[num_cols] = scaler.fit_transform(X[num_cols])

print("\nNumerical features scaled successfully.")

from sklearn.model_selection import train_test_split

# -----------------------------
# TRAIN-TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)

