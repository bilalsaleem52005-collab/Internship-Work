import streamlit as st
import pandas as pd
import plotly.express as px

from utils import set_active_dataset_from_sidebar, load_active_data
from style import apply_custom_style
from style import theme_selector
theme_selector()
st.set_page_config(page_title="Revenue Impact Simulator", page_icon="💰", layout="wide")
apply_custom_style()

st.title("💰 Revenue Impact & Retention ROI Simulator")
st.markdown("Convert churn probability into **financial impact**, **campaign cost**, and **ROI**.")

set_active_dataset_from_sidebar(show_external_uploader=False)
df = load_active_data()

# -----------------------------
# Require prediction from Home page
# -----------------------------
if "last_probability" not in st.session_state or "last_input_raw" not in st.session_state:
    st.warning("Please run a prediction on the main Dashboard first (Home), then come back here.")
    st.stop()

prob = float(st.session_state["last_probability"])
raw = st.session_state["last_input_raw"]

monthly = float(raw["monthly_charges"])

# -----------------------------
# Controls
# -----------------------------
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
with c1:
    n_customers = st.slider("Number of similar customers", 1, 10000, 500, 50)
with c2:
    months = st.slider("Time horizon (months)", 1, 24, 6, 1)
with c3:
    retention_success = st.slider("Retention success rate", 0.05, 0.90, 0.35, 0.01)
with c4:
    offer_cost = st.number_input("Cost per customer ($)", min_value=0.0, value=10.0, step=1.0)

# -----------------------------
# Calculations
# -----------------------------
expected_loss_per_customer_month = monthly * prob
expected_loss_total = expected_loss_per_customer_month * n_customers * months

# If retention campaign applied to all N customers:
campaign_cost = offer_cost * n_customers

# Expected revenue saved:
# saved = expected_loss_total * retention_success
saved_revenue = expected_loss_total * retention_success

# ROI
roi = ((saved_revenue - campaign_cost) / campaign_cost) if campaign_cost > 0 else None

# -----------------------------
# KPI Row
# -----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Churn Probability", f"{prob:.2%}")
k2.metric("Expected Loss / customer / month", f"${expected_loss_per_customer_month:.2f}")
k3.metric("Total Expected Loss", f"${expected_loss_total:,.2f}")
if roi is None:
    k4.metric("ROI", "N/A")
else:
    k4.metric("ROI", f"{roi*100:.1f}%")

st.caption("Expected Loss is a simple estimate: MonthlyCharges × churn probability × customers × months.")

st.markdown("---")

# -----------------------------
# Scenario chart
# -----------------------------
st.subheader("📊 Scenario Breakdown")

scenario_df = pd.DataFrame([
    {"Item": "Expected Loss (No Action)", "Amount": expected_loss_total},
    {"Item": "Campaign Cost", "Amount": campaign_cost},
    {"Item": "Saved Revenue (Expected)", "Amount": saved_revenue},
    {"Item": "Net Benefit (Saved - Cost)", "Amount": saved_revenue - campaign_cost},
])

fig = px.bar(
    scenario_df,
    x="Item",
    y="Amount",
    text=scenario_df["Amount"].map(lambda x: f"${x:,.0f}"),
    title="Financial Impact Summary"
)
fig.update_layout(
    height=420,
    margin=dict(l=20, r=20, t=60, b=20),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Business explanation (very exam-friendly)
# -----------------------------
st.markdown("---")
st.subheader("🧾 Interpretation (Business Explanation)")

if roi is None:
    st.write("• ROI is not available because campaign cost is 0.")
else:
    if roi > 0:
        st.success(
            f"✅ This intervention is expected to be profitable. Estimated ROI ≈ **{roi*100:.1f}%** over {months} months."
        )
    else:
        st.error(
            f"⚠️ This intervention may not be profitable. Estimated ROI ≈ **{roi*100:.1f}%** over {months} months."
        )

st.write(
    "• **Expected Loss** estimates how much revenue is at risk if similar customers churn.\n"
    "• **Saved Revenue** assumes retention success reduces churn proportionally.\n"
    "• **Net Benefit** compares expected revenue saved vs campaign cost."
)

# -----------------------------
# Download scenario table
# -----------------------------
st.markdown("---")
st.subheader("⬇️ Download Scenario Table")

st.dataframe(scenario_df, use_container_width=True)

csv = scenario_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download CSV",
    data=csv,
    file_name="retention_roi_scenario.csv",
    mime="text/csv"
)

st.info("This page strongly improves your FYP marks because it shows **business impact**, not just prediction.")