# safe_load_model.py
# This script ONLY READS the file — it cannot damage your .pkl

import pickle
import os
import xgboost as xgb   # pip install xgboost if missing

# Adjust this path if your file is not inside a "models" folder
MODEL_PATH = "models/xgboost_churn_model.pkl"

# Step 1: Basic safety checks
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: File not found at → {MODEL_PATH}")
    print("Current folder contents:")
    print(os.listdir(os.path.dirname(MODEL_PATH) or "."))
    exit()

file_size = os.path.getsize(MODEL_PATH)
print(f"File found! Size = {file_size:,} bytes (~{file_size / 1024 / 1024:.2f} MB)")

# Step 2: Try to load (read-only)
try:
    with open(MODEL_PATH, "rb") as f:          # "rb" = read binary → very important
        model = pickle.load(f)
    
    print("\nSUCCESS! Model loaded safely.")
    print("Model type:     ", type(model).__name__)
    print("Is XGBoost?     ", isinstance(model, xgb.XGBClassifier))
    print("Number of trees:", model.n_estimators if hasattr(model, 'n_estimators') else "N/A")
    print("Objective:      ", model.objective if hasattr(model, 'objective') else "N/A")
    
    # Optional: show first few important features (if model was trained)
    if hasattr(model, "feature_importances_"):
        print("\nTop 5 feature importances:")
        importances = model.feature_importances_
        indices = importances.argsort()[-5:][::-1]
        for i in indices:
            print(f"  {i:3d} → {importances[i]:.4f}")

except Exception as e:
    print("\nLOAD FAILED — but your file is still safe!")
    print("Error message:", str(e))
    print("\nCommon fixes:")
    print("• Make sure xgboost is installed:  pip install xgboost")
    print("• Try with joblib instead (some models are saved that way):")
    print("    from joblib import load")
    print("    model = load(MODEL_PATH)")
    print("• Wrong Python/xgboost version? Use the same version you trained with.")

print("\nDone. Your original file is unchanged.")