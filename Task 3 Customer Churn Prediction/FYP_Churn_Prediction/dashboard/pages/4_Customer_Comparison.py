import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import set_active_dataset_from_sidebar, load_active_data
from style import apply_custom_style
from style import theme_selector
theme_selector()
from utils import load_telco_data, require_prediction

st.set_page_config(page_title="Customer Comparison", page_icon="📈", layout="wide")
apply_custom_style()

st.title("📈 Customer vs Dataset Comparison")
st.markdown("Compare this customer to the overall dataset using **benchmarks + percentiles**.")

set_active_dataset_from_sidebar(show_external_uploader=False)
df = load_active_data()

df = load_telco_data()
probability, raw = require_prediction()

cust_tenure = raw["tenure"]
cust_monthly = raw["monthly_charges"]
cust_total = raw["total_charges"]

# --- Percentiles
tenure_pct = (df["tenure"].rank(pct=True) * 100).loc[df["tenure"].sub(cust_tenure).abs().idxmin()]
monthly_pct = (df["MonthlyCharges"].rank(pct=True) * 100).loc[df["MonthlyCharges"].sub(cust_monthly).abs().idxmin()]
total_pct = (df["TotalCharges"].rank(pct=True) * 100).loc[df["TotalCharges"].sub(cust_total).abs().idxmin()]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Predicted Risk", f"{probability:.2%}")
k2.metric("Tenure", f"{cust_tenure} months")
k3.metric("Monthly Charges", f"${cust_monthly:.2f}")
k4.metric("Total Charges", f"${cust_total:.2f}")

st.markdown("---")

# --- Radar chart (different type)
labels = ["Tenure", "Monthly Charges", "Total Charges"]
cust_vals = [cust_tenure, cust_monthly, cust_total]
avg_vals = [df["tenure"].mean(), df["MonthlyCharges"].mean(), df["TotalCharges"].mean()]

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(r=cust_vals, theta=labels, fill="toself", name="Customer"))
fig_radar.add_trace(go.Scatterpolar(r=avg_vals, theta=labels, fill="toself", name="Dataset Avg"))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True)),
    title="Radar Comparison: Customer vs Dataset Avg",
    height=420,
    margin=dict(l=20, r=20, t=60, b=20),
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_radar, use_container_width=True)

# --- Percentile tiles (simple, very “business”)
st.subheader("🏁 Percentile Position (How extreme is this customer?)")
p1, p2, p3 = st.columns(3)
p1.metric("Tenure Percentile", f"{tenure_pct:.0f}th")
p2.metric("Monthly Charges Percentile", f"{monthly_pct:.0f}th")
p3.metric("Total Charges Percentile", f"{total_pct:.0f}th")

st.caption("Example: 85th percentile monthly charges means this customer pays more than ~85% of customers.")

st.markdown("---")

# --- Distribution overlay (hist + marker)
fig_hist = px.histogram(df, x="MonthlyCharges", nbins=40, title="Monthly Charges Distribution (Customer marked)")
fig_hist.add_vline(x=cust_monthly, line_width=3, line_dash="dash")
fig_hist.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_hist, use_container_width=True)

# --- Violin plot (different chart type)
fig_violin = px.violin(df, y="MonthlyCharges", box=True, points="suspectedoutliers", title="Monthly Charges Spread (Violin + Box)")
fig_violin.add_hline(y=cust_monthly, line_width=3, line_dash="dash")
fig_violin.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_violin, use_container_width=True)