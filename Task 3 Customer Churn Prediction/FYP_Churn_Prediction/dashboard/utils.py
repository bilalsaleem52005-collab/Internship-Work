import os
import re
import pandas as pd
import streamlit as st

# ---------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------

def get_project_root():
    # dashboard/utils.py -> dashboard -> project root
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_data_path(filename="telco_churn.csv"):
    return os.path.join(get_project_root(), "data", filename)

def get_model_path(filename="xgboost_churn_model.pkl"):
    return os.path.join(get_project_root(), "models", filename)


# ---------------------------------------------------
# ORIGINAL DATA LOADER (BUILT-IN TELCO)
# ---------------------------------------------------

@st.cache_data
def load_telco_data():
    data_path = get_data_path("telco_churn.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at:\n{data_path}\n\n"
            "Expected location:\n<project_root>/data/telco_churn.csv"
        )

    df = pd.read_csv(data_path)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = df.dropna().copy()
    return df


# ---------------------------------------------------
# PREDICTION CHECK (USED BY OTHER PAGES)
# ---------------------------------------------------

def require_prediction():
    """
    Returns (probability, raw_inputs_dict).
    Stops the page nicely if prediction doesn't exist.
    """
    if "last_probability" not in st.session_state or "last_input_raw" not in st.session_state:
        st.warning("⚠️ Please make a prediction first from the main Dashboard page.")
        st.stop()
    return float(st.session_state["last_probability"]), st.session_state["last_input_raw"]


# ===================================================
# DATASET SWITCHER + EXTERNAL VALIDATION SUPPORT
# ===================================================

# ---------------------------------------------------
# COLUMN NORMALIZATION
# ---------------------------------------------------

COLUMN_MAP = {
    "Monthly Charge": "MonthlyCharges",
    "Monthly_Charges": "MonthlyCharges",
    "Total Charge": "TotalCharges",
    "Total_Charges": "TotalCharges",
    "Internet Service": "InternetService",
    "Payment Method": "PaymentMethod",
    "Senior Citizen": "SeniorCitizen",
    "Dependents?": "Dependents",
}

def normalize_columns(df: pd.DataFrame):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

    def canon(x):
        return re.sub(r"[\s_]+", "", x.lower())

    canon_map = {canon(c): c for c in df.columns}

    for expected in [
        "MonthlyCharges",
        "TotalCharges",
        "InternetService",
        "PaymentMethod",
        "SeniorCitizen",
    ]:
        key = canon(expected)
        if key in canon_map and canon_map[key] != expected:
            df = df.rename(columns={canon_map[key]: expected})

    return df


# ---------------------------------------------------
# SCHEMA VALIDATION
# ---------------------------------------------------

REQUIRED_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
    "PaymentMethod",
    "Churn",
]

def validate_external_schema(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing


# ---------------------------------------------------
# SIDEBAR DATASET SWITCHER
# ---------------------------------------------------

def set_active_dataset_from_sidebar(show_external_uploader: bool = False):
    """
    - Dataset selectbox shown on ALL pages.
    - External uploader shown ONLY when show_external_uploader=True (Dashboard).
    """

    # session defaults
    if "dataset_mode" not in st.session_state:
        st.session_state["dataset_mode"] = "Telco (Built-in)"

    if "external_df" not in st.session_state:
        st.session_state["external_df"] = None

    if "external_sig" not in st.session_state:
        st.session_state["external_sig"] = None

    if "prev_dataset_mode" not in st.session_state:
        st.session_state["prev_dataset_mode"] = st.session_state["dataset_mode"]

    with st.sidebar:
        st.markdown("### 🗂️ Dataset")

        mode = st.selectbox(
            "Select dataset",
            ["Telco (Built-in)", "External CSV"],
            index=0 if st.session_state["dataset_mode"] == "Telco (Built-in)" else 1,
            key="dataset_mode_selectbox_unique"
        )

        st.session_state["dataset_mode"] = mode

        # Reset prediction/row if dataset changed
        if st.session_state["prev_dataset_mode"] != st.session_state["dataset_mode"]:
            st.session_state.pop("last_probability", None)
            st.session_state.pop("last_input_raw", None)
            st.session_state.pop("selected_customer_row", None)
            st.session_state.pop("dataset_row_index", None)
            st.session_state["prev_dataset_mode"] = st.session_state["dataset_mode"]

        # External CSV (ONLY show uploader on Dashboard)
        if st.session_state["dataset_mode"] == "External CSV":

            # other pages: no uploader
            if not show_external_uploader:
                if st.session_state.get("external_df") is not None:
                    st.success("External dataset loaded ✅")
                else:
                    st.warning("External dataset not loaded. Upload it on Dashboard (app).")
                return

            # Dashboard: show uploader
            uploaded = st.file_uploader(
                "Upload validation dataset (CSV)",
                type=["csv"],
                key="external_csv_uploader_unique"
            )

            if uploaded is None:
                st.info("Upload a CSV to use as external dataset.")
                return

            file_sig = (uploaded.name, uploaded.size)

            # reload only if new file
            if st.session_state.get("external_sig") != file_sig:
                df = pd.read_csv(uploaded)
                df = normalize_columns(df)

                missing = validate_external_schema(df)
                if missing:
                    st.session_state["external_df"] = None
                    st.session_state["external_sig"] = file_sig
                    st.error(f"Dataset missing required columns: {missing}")
                    return

                st.session_state["external_df"] = df
                st.session_state["external_sig"] = file_sig
                st.success(f"External dataset loaded ✅ ({uploaded.name})")

            # ✅ NO debug table, NO value_counts, nothing.


# ---------------------------------------------------
# LOAD CURRENTLY ACTIVE DATASET (USED IN ALL PAGES)
# ---------------------------------------------------

def load_active_data():
    mode = st.session_state.get("dataset_mode", "Telco (Built-in)")

    if mode == "External CSV":
        df = st.session_state.get("external_df", None)
        if df is None:
            st.warning("External dataset not loaded yet. Upload it on Dashboard (app).")
            st.stop()
        return df.copy()

    return load_telco_data()


# ---------------------------------------------------
# PREPROCESS DATA FOR MODEL EVALUATION
# ---------------------------------------------------

def preprocess_for_model(df: pd.DataFrame, feature_names):
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    if "Churn" not in df.columns:
        raise ValueError("Dataset must contain 'Churn' column.")

    y = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)

    drop_cols = ["customerID", "Churn"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    obj_cols = X.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        X = pd.get_dummies(X, columns=obj_cols)

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_names]
    return X, y
