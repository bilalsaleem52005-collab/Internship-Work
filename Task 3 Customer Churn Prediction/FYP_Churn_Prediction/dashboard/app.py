import os
import pickle
import pandas as pd
import streamlit as st
import shap

from utils import set_active_dataset_from_sidebar, load_active_data

from style import apply_theme
from ui import hr, section_title

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📉",
    layout="wide"
)

st.caption(f"Dataset in use: {st.session_state.get('dataset_mode','Telco (Built-in)')}")

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📉",
    layout="wide"
)

# Dataset switcher (works across all pages)

set_active_dataset_from_sidebar(show_external_uploader=True)
df = load_active_data()


# NEW: choose prediction source


# Theme toggle
with st.sidebar:
    st.markdown("### 🎨 Theme")
    theme = st.radio("Mode", ["Light", "Dark"], index=0)
apply_theme(theme)

# -----------------------------
# LOAD MODEL (SAFE PATH)
# -----------------------------
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_churn_model.pkl")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

# -----------------------------
# HEADER (YOUR ORIGINAL)
# -----------------------------
st.title("📊 Customer Churn Prediction System")
st.markdown("This system predicts whether a customer is likely to **churn** and provides **actionable recommendations**.")

hr()

# -----------------------------
# SIDEBAR INPUTS (YOUR ORIGINAL)
# -----------------------------
st.sidebar.header("🧾 Customer Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner?", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has Dependents?", ["No", "Yes"])

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 20, 120, 70)
total_charges = st.sidebar.slider("Total Charges ($)", 0, 9000, 1000)

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes"])
payment = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

def build_input():

    # If user picked a dataset row, use it
    row = st.session_state.get("selected_customer_row")

    if row:
        # Map dataset row -> your variables (fallback to existing sidebar values if missing)
        _gender = str(row.get("gender", gender))
        _senior = "Yes" if str(row.get("SeniorCitizen", 0)) in ["1", "1.0", "Yes", "yes", "True", "true"] else "No"
        _partner = str(row.get("Partner", partner))
        _dependents = str(row.get("Dependents", dependents))

        _tenure = int(float(row.get("tenure", tenure)))
        _monthly = float(row.get("MonthlyCharges", monthly_charges))
        _total = float(row.get("TotalCharges", total_charges))

        _contract = str(row.get("Contract", contract))
        _internet = str(row.get("InternetService", internet))
        _tech = str(row.get("TechSupport", tech_support))
        _security = str(row.get("OnlineSecurity", online_security))
        _payment = str(row.get("PaymentMethod", payment))

    else:
        # Manual sidebar values
        _gender = gender
        _senior = senior
        _partner = partner
        _dependents = dependents
        _tenure = tenure
        _monthly = monthly_charges
        _total = total_charges
        _contract = contract
        _internet = internet
        _tech = tech_support
        _security = online_security
        _payment = payment

    data = {
        "SeniorCitizen": 1 if _senior == "Yes" else 0,
        "tenure": _tenure,
        "MonthlyCharges": _monthly,
        "TotalCharges": _total,
        "gender_Male": 1 if _gender == "Male" else 0,
        "Partner_Yes": 1 if str(_partner).strip().lower() == "yes" else 0,
        "Dependents_Yes": 1 if str(_dependents).strip().lower() == "yes" else 0,
        "InternetService_Fiber optic": 1 if _internet == "Fiber optic" else 0,
        "InternetService_No": 1 if _internet == "No" else 0,
        "TechSupport_Yes": 1 if str(_tech).strip().lower() == "yes" else 0,
        "OnlineSecurity_Yes": 1 if str(_security).strip().lower() == "yes" else 0,
        "Contract_One year": 1 if _contract == "One year" else 0,
        "Contract_Two year": 1 if _contract == "Two year" else 0,
        "PaymentMethod_Electronic check": 1 if _payment == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if _payment == "Mailed check" else 0,
    }

    return pd.DataFrame([data])

# -----------------------------
# PREDICTION (CLEAN)
# -----------------------------
section_title("🔍 Prediction Result", "Click predict to generate churn risk + insights.")
predict_clicked = st.button("🚀 Predict Churn")

if predict_clicked:
    input_df = build_input()

    for col in model.get_booster().feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.get_booster().feature_names]

    probability = float(model.predict_proba(input_df)[0][1])

    st.session_state["last_probability"] = probability
    st.session_state["last_input"] = input_df.copy()
    st.session_state["last_input_raw"] = {
        "gender": gender,
        "senior": senior,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract": contract,
        "internet": internet,
        "tech_support": tech_support,
        "online_security": online_security,
        "payment": payment,
    }

    prediction = "YES" if probability >= 0.5 else "NO"

    if probability >= 0.7:
        risk_level = "High Risk"
        risk_icon = "🔴"
    elif probability >= 0.4:
        risk_level = "Medium Risk"
        risk_icon = "🟡"
    else:
        risk_level = "Low Risk"
        risk_icon = "🟢"

    expected_loss = monthly_charges * probability

    # Minimal KPI row (NO tiles/cubes)
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Churn Probability", f"{probability:.2%}")
    with k2:
        st.metric("Risk Level", f"{risk_icon} {risk_level}")
    with k3:
        st.metric("Expected Monthly Revenue at Risk", f"${expected_loss:.2f}")

    hr()

    # Outcome line (clean)
    if prediction == "YES":
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")

    hr()

    # SHAP EXPLANATION
    section_title("🧠 Why This Prediction?", "Top features pushing the prediction.")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    shap_series = pd.Series(shap_values[0], index=input_df.columns).sort_values(ascending=False)
    churn_drivers = shap_series[shap_series > 0].head(3)
    stay_drivers = shap_series[shap_series < 0].tail(3)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🔴 Factors Increasing Churn Risk**")
        if len(churn_drivers) > 0:
            for f in churn_drivers.index:
                st.write("•", f)
        else:
            st.write("No strong churn drivers detected.")
    with c2:
        st.markdown("**🟢 Factors Supporting Retention**")
        if len(stay_drivers) > 0:
            for f in stay_drivers.index:
                st.write("•", f)
        else:
            st.write("No strong retention drivers detected.")

    hr()

    # RECOMMENDATIONS
    section_title("💡 Recommended Retention Actions", "Actions generated from the customer profile.")
    recommendations = []
    if contract == "Month-to-month":
        recommendations.append("📄 Offer a discounted long-term contract")
    if monthly_charges > 80:
        recommendations.append("💸 Provide a temporary price reduction or loyalty discount")
    if tech_support == "No":
        recommendations.append("🛠️ Offer free tech support for 3 months")
    if online_security == "No":
        recommendations.append("🔐 Bundle free online security service")
    if tenure < 12:
        recommendations.append("🎁 Give onboarding rewards for new customers")
    if not recommendations:
        recommendations.append("👍 Maintain current service quality and engagement")

    for r in recommendations:
        st.write("•", r)

else:
    st.markdown("<div class='muted'>Set the customer inputs in the sidebar, then click <b>Predict Churn</b>.</div>", unsafe_allow_html=True)

hr()
st.caption("Final Year Project | AI-Based Customer Churn Prediction System")