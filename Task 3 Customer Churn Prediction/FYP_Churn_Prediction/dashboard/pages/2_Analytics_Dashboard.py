import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import set_active_dataset_from_sidebar, load_active_data
from style import apply_theme

with st.sidebar:
    st.session_state["theme"] = st.radio("Theme", ["Light", "Dark"], index=0 if st.session_state.get("theme","Light")=="Light" else 1)

apply_theme(st.session_state["theme"])
# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)
 
set_active_dataset_from_sidebar(show_external_uploader=False)
df = load_active_data()

st.title("📊 Customer Churn Analytics Dashboard")
st.markdown(
    "This page provides **business insights and analytics** to understand customer churn patterns "
    "for **customers similar to the one you just predicted**."
)

# -----------------------------
# LOAD DATA (FIXED PATH)
# -----------------------------
@st.cache_data
def load_data():
    # pages -> dashboard -> project root
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "telco_churn.csv")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at:\n{DATA_PATH}\n\n"
            "Fix: Make sure telco_churn.csv is inside:\n"
            "<project_root>/data/telco_churn.csv"
        )

    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    return df

df = load_data()

# -----------------------------
# CHECK IF PREDICTION EXISTS
# -----------------------------
if "last_input" not in st.session_state or "last_probability" not in st.session_state:
    st.warning("⚠️ Please make a prediction first from the main dashboard (app.py).")
    st.stop()

input_df = st.session_state["last_input"]        # encoded + aligned columns used for model
probability = float(st.session_state["last_probability"])

# -----------------------------
# CREATE SIMILAR CUSTOMER FILTER (DYNAMIC)
# -----------------------------
filtered_df = df.copy()

# We use encoded input_df values because that's what you stored.
# But we filter the ORIGINAL dataset df (human-readable columns).

# Tenure logic
try:
    tenure_val = float(input_df["tenure"].values[0])
    if tenure_val < 12:
        filtered_df = filtered_df[filtered_df["tenure"] < 12]
    elif tenure_val < 24:
        filtered_df = filtered_df[(filtered_df["tenure"] >= 12) & (filtered_df["tenure"] < 24)]
    else:
        filtered_df = filtered_df[filtered_df["tenure"] >= 24]
except:
    pass

# Monthly charges logic
try:
    mc_val = float(input_df["MonthlyCharges"].values[0])
    if mc_val > 80:
        filtered_df = filtered_df[filtered_df["MonthlyCharges"] > 80]
    elif mc_val > 50:
        filtered_df = filtered_df[(filtered_df["MonthlyCharges"] >= 50) & (filtered_df["MonthlyCharges"] <= 80)]
    else:
        filtered_df = filtered_df[filtered_df["MonthlyCharges"] < 50]
except:
    pass

# Contract logic (Month-to-month vs One/Two year)
# In your encoding (drop_first=True), "Contract_One year" and "Contract_Two year" exist.
try:
    one_year = int(input_df.get("Contract_One year", pd.Series([0])).values[0])
    two_year = int(input_df.get("Contract_Two year", pd.Series([0])).values[0])

    if one_year == 1:
        filtered_df = filtered_df[filtered_df["Contract"] == "One year"]
    elif two_year == 1:
        filtered_df = filtered_df[filtered_df["Contract"] == "Two year"]
    else:
        filtered_df = filtered_df[filtered_df["Contract"] == "Month-to-month"]
except:
    pass

# Tech support logic
try:
    tech_yes = int(input_df.get("TechSupport_Yes", pd.Series([0])).values[0])
    if tech_yes == 1:
        filtered_df = filtered_df[filtered_df["TechSupport"] == "Yes"]
    else:
        filtered_df = filtered_df[filtered_df["TechSupport"] == "No"]
except:
    pass

# Internet service logic
# In your encoding, Fiber optic and No exist (DSL is baseline).
try:
    fiber = int(input_df.get("InternetService_Fiber optic", pd.Series([0])).values[0])
    no_net = int(input_df.get("InternetService_No", pd.Series([0])).values[0])

    if no_net == 1:
        filtered_df = filtered_df[filtered_df["InternetService"] == "No"]
    elif fiber == 1:
        filtered_df = filtered_df[filtered_df["InternetService"] == "Fiber optic"]
    else:
        filtered_df = filtered_df[filtered_df["InternetService"] == "DSL"]
except:
    pass

# If filtering becomes too strict, fallback
if len(filtered_df) < 30:
    filtered_df = df.copy()

# -----------------------------
# KPI SECTION
# -----------------------------
st.markdown("## 📌 Dynamic Risk Analysis (Based on Your Prediction)")

total_similar = len(filtered_df)
similar_churned = len(filtered_df[filtered_df["Churn"] == "Yes"])
similar_churn_rate = similar_churned / total_similar if total_similar > 0 else 0

overall_churn_rate = (df["Churn"] == "Yes").mean()

col1, col2, col3 = st.columns(3)
col1.metric("Predicted Customer Risk", f"{probability:.2%}")
col2.metric("Similar Customers Churn Rate", f"{similar_churn_rate:.2%}")
col3.metric("Overall Dataset Churn Rate", f"{overall_churn_rate:.2%}")

st.caption(f"Similar group size used for analytics: **{total_similar} customers**")
st.markdown("---")

# -----------------------------
# CONTRACT TYPE ANALYSIS
# -----------------------------
st.subheader("📄 Churn by Contract Type (Similar Customers)")

contract_churn = pd.crosstab(filtered_df["Contract"], filtered_df["Churn"])

fig1, ax1 = plt.subplots(figsize=(8, 4))
contract_churn.plot(kind="bar", ax=ax1)
plt.xticks(rotation=0)
plt.ylabel("Number of Customers")
plt.title("Churn Distribution by Contract Type")
st.pyplot(fig1)

st.markdown(
"""
**Insight:**  
Customers on *Month-to-Month* contracts usually show higher churn.  
Encouraging long-term contracts often reduces churn.
"""
)

st.markdown("---")

# -----------------------------
# TENURE ANALYSIS
# -----------------------------
st.subheader("⏳ Tenure vs Churn (Similar Customers)")

fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.histplot(data=filtered_df, x="tenure", hue="Churn", bins=30, kde=True, ax=ax2)
plt.title("Customer Tenure Distribution by Churn")
st.pyplot(fig2)

st.markdown(
"""
**Insight:**  
Low tenure (new customers) tends to have higher churn.  
Better onboarding + early support can reduce churn.
"""
)

st.markdown("---")

# -----------------------------
# MONTHLY CHARGES ANALYSIS
# -----------------------------
st.subheader("💰 Monthly Charges vs Churn (Similar Customers)")

fig3, ax3 = plt.subplots(figsize=(8, 4))
sns.boxplot(data=filtered_df, x="Churn", y="MonthlyCharges", ax=ax3)
plt.title("Monthly Charges Distribution by Churn")
st.pyplot(fig3)

st.markdown(
"""
**Insight:**  
Higher monthly charges often increase churn risk.  
Discounts, bundles, and loyalty rewards can help retain customers.
"""
)

st.markdown("---")

# -----------------------------
# INTERNET SERVICE ANALYSIS
# -----------------------------
st.subheader("🌐 Internet Service Impact (Similar Customers)")

internet_churn = pd.crosstab(filtered_df["InternetService"], filtered_df["Churn"])

fig4, ax4 = plt.subplots(figsize=(8, 4))
internet_churn.plot(kind="bar", ax=ax4)
plt.xticks(rotation=0)
plt.ylabel("Number of Customers")
plt.title("Churn by Internet Service Type")
st.pyplot(fig4)

st.markdown(
"""
**Insight:**  
Fiber optic customers can show higher churn (often due to price or service issues).  
Improving service quality + targeted retention offers can reduce churn.
"""
)

st.markdown("---")

# -----------------------------
# SUMMARY SECTION
# -----------------------------
st.subheader("📌 Executive Summary (Auto)")

st.info(
f"""
**Your Customer Risk:** {probability:.2%}  
**Similar Group Churn Rate:** {similar_churn_rate:.2%}  
**Overall Dataset Churn Rate:** {overall_churn_rate:.2%}  

Key Drivers often linked with churn (in similar customers):
• Month-to-Month Contract  
• High Monthly Charges  
• Short Tenure  
• Fiber Optic Internet Users  

Recommended Strategy:
• Promote Long-Term Contracts  
• Introduce Loyalty Discounts  
• Improve Early Customer Experience  
• Provide Value-Added Service Bundles  
"""
)

st.markdown("---")
st.caption("Final Year Project | AI-Based Customer Churn Prediction System")