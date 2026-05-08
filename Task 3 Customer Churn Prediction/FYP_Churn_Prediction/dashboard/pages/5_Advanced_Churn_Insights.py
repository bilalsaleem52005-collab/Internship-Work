import streamlit as st
import pandas as pd
import plotly.express as px

from utils import set_active_dataset_from_sidebar, load_active_data
from style import apply_custom_style
from utils import load_telco_data, require_prediction
from style import theme_selector
theme_selector()

st.set_page_config(page_title="Advanced Churn Insights", page_icon="🔥", layout="wide")
apply_custom_style()

st.title("🔥 Advanced Churn Insights")
st.markdown("Deep analytics and explainable patterns for stakeholders.")

set_active_dataset_from_sidebar(show_external_uploader=False)
df = load_active_data()

df = load_telco_data()
probability, raw = require_prediction()

# --- Sunburst (existing idea)
st.subheader("🌐 Churn Breakdown (Interactive)")
fig = px.sunburst(
    df,
    path=["Contract", "InternetService", "PaymentMethod", "Churn"],
    title="Churn Breakdown (Contract → Internet → Payment → Churn)"
)
fig.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- Heatmap (different chart type)
st.subheader("🧩 Churn Heatmap (Contract × Internet)")
pivot = df.pivot_table(
    index="Contract",
    columns="InternetService",
    values="Churn",
    aggfunc=lambda x: (x == "Yes").mean()
)
pivot = pivot.fillna(0)

fig_hm = px.imshow(
    pivot,
    text_auto=".1%",
    aspect="auto",
    title="Churn Rate Heatmap"
)
fig_hm.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_hm, use_container_width=True)

st.markdown("---")

# --- Payment method ranking
st.subheader("💳 Payment Method Churn Ranking")
pay = df.groupby("PaymentMethod").agg(
    ChurnRate=("Churn", lambda x: (x == "Yes").mean()),
    Count=("Churn", "size")
).reset_index().sort_values("ChurnRate", ascending=False)

fig_pay = px.bar(
    pay, x="PaymentMethod", y="ChurnRate",
    text=pay["ChurnRate"].map(lambda v: f"{v:.1%}"),
    title="Churn Rate by Payment Method"
)
fig_pay.update_layout(height=380, yaxis=dict(tickformat=".0%"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_pay, use_container_width=True)

st.markdown("---")

# --- What-if simulator (simple but very impressive)
st.subheader("🧪 Quick What-If Simulator (Business Demo)")
st.caption("Adjust key factors and see how churn rates shift in the dataset (pattern simulation, not re-prediction).")

sim_contract = st.selectbox("Contract scenario", ["Month-to-month", "One year", "Two year"], index=0)
sim_internet = st.selectbox("Internet scenario", ["DSL", "Fiber optic", "No"], index=0)
sim_payment = st.selectbox("Payment scenario", df["PaymentMethod"].unique().tolist(), index=0)

subset = df[(df["Contract"] == sim_contract) & (df["InternetService"] == sim_internet) & (df["PaymentMethod"] == sim_payment)]
if len(subset) < 30:
    st.warning("Segment too small, showing broader segment: Contract × Internet.")
    subset = df[(df["Contract"] == sim_contract) & (df["InternetService"] == sim_internet)]

rate = (subset["Churn"] == "Yes").mean()
st.metric("Estimated churn rate for selected segment", f"{rate:.1%}")