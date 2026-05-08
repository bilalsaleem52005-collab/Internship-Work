import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
from utils import set_active_dataset_from_sidebar, load_active_data
from style import theme_selector
theme_selector()
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc,
    confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

from style import apply_custom_style


st.set_page_config(page_title="Model Performance", page_icon="📊", layout="wide")
apply_custom_style()

st.title("📊 Model Performance & Evaluation")
st.markdown("This page evaluates the saved **XGBoost churn model** using a clean holdout test split.")
set_active_dataset_from_sidebar(show_external_uploader=False)
df = load_active_data()
# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():

    # current file → dashboard/pages/8_Model_Performance.py
    pages_dir = os.path.dirname(os.path.abspath(__file__))

    # dashboard folder
    dashboard_dir = os.path.dirname(pages_dir)

    # project root folder
    project_root = os.path.dirname(dashboard_dir)

    # correct model path
    model_path = os.path.join(project_root, "models", "xgboost_churn_model.pkl")

    if not os.path.exists(model_path):
        st.error(f"Model not found at: {model_path}")
        st.stop()

    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():

    # current file → dashboard/pages/8_Model_Performance.py
    pages_dir = os.path.dirname(os.path.abspath(__file__))

    # dashboard folder
    dashboard_dir = os.path.dirname(pages_dir)

    # project root folder
    project_root = os.path.dirname(dashboard_dir)

    data_path = os.path.join(project_root, "data", "telco_churn.csv")

    if not os.path.exists(data_path):
        st.error(f"Dataset not found at: {data_path}")
        st.stop()

    return pd.read_csv(data_path)

df = load_data()

# -----------------------------
# Preprocess to match model features
# -----------------------------
@st.cache_data
def preprocess_for_model(df: pd.DataFrame, feature_names: list[str]):
    df = df.copy()

    # Clean TotalCharges (Telco dataset has blanks)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Target
    if "Churn" not in df.columns:
        raise ValueError("Dataset must contain 'Churn' column.")
    y = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)

    # Drop non-features
    drop_cols = [c for c in ["customerID", "Churn"] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")

    # One-hot encode all object columns
    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        X = pd.get_dummies(X, columns=obj_cols, drop_first=False)

    # Align with model feature space
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    # Keep only model columns in same order
    X = X[feature_names]

    return X, y

feature_names = model.get_booster().feature_names
X, y = preprocess_for_model(df, feature_names)

# -----------------------------
# Controls
# -----------------------------
st.markdown("---")
c1, c2, c3 = st.columns([1.1, 1.1, 1.0])
with c1:
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
with c2:
    random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
with c3:
    threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.01)

# -----------------------------
# Split & Predict
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_state), stratify=y
)

proba = model.predict_proba(X_test)[:, 1]
y_pred = (proba >= threshold).astype(int)

# -----------------------------
# Metrics
# -----------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

fpr, tpr, _ = roc_curve(y_test, proba)
roc_auc = auc(fpr, tpr)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Accuracy", f"{acc:.3f}")
k2.metric("Precision", f"{prec:.3f}")
k3.metric("Recall", f"{rec:.3f}")
k4.metric("F1-score", f"{f1:.3f}")
k5.metric("ROC-AUC", f"{roc_auc:.3f}")

st.caption("Tip: move the threshold slider to show how trade-offs change (precision vs recall).")

st.markdown("---")

# -----------------------------
# ROC Curve (Plotly)
# -----------------------------
roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})"))
roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
roc_fig.update_layout(
    title="ROC Curve",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    height=420,
    margin=dict(l=20, r=20, t=60, b=20),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(roc_fig, use_container_width=True)

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=["Actual Stay", "Actual Churn"], columns=["Pred Stay", "Pred Churn"])

cm_fig = px.imshow(
    cm_df,
    text_auto=True,
    aspect="auto",
    title="Confusion Matrix",
)
cm_fig.update_layout(
    height=380,
    margin=dict(l=20, r=20, t=60, b=20),
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(cm_fig, use_container_width=True)

# -----------------------------
# Feature Importance (XGBoost built-in)
# -----------------------------
st.markdown("---")
st.subheader("⭐ Feature Importance (Model Explainability)")

# Safer: use feature importance from booster
score = model.get_booster().get_score(importance_type="gain")  # gain is usually best
if not score:
    st.warning("Feature importance not available from model booster.")
else:
    imp = (
        pd.DataFrame({"feature": list(score.keys()), "gain": list(score.values())})
        .sort_values("gain", ascending=False)
        .head(20)
    )
    fig_imp = px.bar(
        imp,
        x="gain",
        y="feature",
        orientation="h",
        title="Top 20 Features by Gain",
    )
    fig_imp.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

st.info(
    "Why this page matters: it shows evaluation metrics + model behavior, which is essential for a real AI system (not just a demo)."
)