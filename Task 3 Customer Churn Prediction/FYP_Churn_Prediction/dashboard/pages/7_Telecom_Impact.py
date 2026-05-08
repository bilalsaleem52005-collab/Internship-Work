import streamlit as st
import pandas as pd
import plotly.express as px

from style import apply_custom_style
from style import theme_selector
theme_selector()
from utils import load_telco_data, require_prediction

from utils import set_active_dataset_from_sidebar, load_active_data

st.set_page_config(page_title="Telecom Impact", page_icon="📡", layout="wide")
apply_custom_style()

st.title("📡 Telecom Service Impact Analysis")
st.markdown("Show how telecom services affect **cost** and **churn risk**.")

set_active_dataset_from_sidebar(show_external_uploader=False)
df = load_active_data()

df = load_telco_data()
probability, raw = require_prediction()

# KPI
k1, k2, k3 = st.columns(3)
k1.metric("Predicted Risk", f"{probability:.2%}")
k2.metric("Customer Internet", raw["internet"])
k3.metric("Customer Monthly Charges", f"${raw['monthly_charges']:.2f}")

st.markdown("---")

# 1) Scatter (different type)
st.subheader("🔎 Tenure vs Charges (colored by churn)")
fig_sc = px.scatter(
    df,
    x="tenure",
    y="MonthlyCharges",
    color="Churn",
    opacity=0.6,
    title="Customer Lifetime vs Cost (Churn pattern)"
)
fig_sc.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_sc, use_container_width=True)

st.markdown("---")

# 2) Service impact
st.subheader("🌐 Impact by Internet Service")

impact = df.groupby("InternetService").agg(
    AvgMonthly=("MonthlyCharges", "mean"),
    ChurnRate=("Churn", lambda x: (x == "Yes").mean()),
    Count=("Churn", "size")
).reset_index()

fig1 = px.bar(impact, x="InternetService", y="AvgMonthly", title="Average Monthly Charges by Internet Service")
fig1.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.bar(impact, x="InternetService", y="ChurnRate", title="Churn Rate by Internet Service", text=impact["ChurnRate"].map(lambda v: f"{v:.1%}"))
fig2.update_layout(height=320, yaxis=dict(tickformat=".0%"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# 3) Contract × Internet top risk segments table (business)
st.subheader("🏷️ Highest Risk Segments (Contract × Internet)")
seg = df.groupby(["Contract", "InternetService"]).agg(
    ChurnRate=("Churn", lambda x: (x == "Yes").mean()),
    AvgMonthly=("MonthlyCharges", "mean"),
    Count=("Churn", "size")
).reset_index().sort_values("ChurnRate", ascending=False)

st.dataframe(seg.head(10), use_container_width=True)

st.info("Use this table to justify retention targeting strategies (which segment to focus first).")