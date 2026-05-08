import streamlit as st
import plotly.express as px
from style import apply_custom_style
from utils import require_prediction
from style import apply_theme
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils import set_active_dataset_from_sidebar, load_active_data
from style import theme_selector
theme_selector()
from utils import load_telco_data, require_prediction

st.set_page_config(page_title="Customer Sentiment", page_icon="💬", layout="wide")
apply_custom_style()

st.title("💬 Customer Sentiment Intelligence")
st.markdown("Translate churn risk into **sentiment & engagement** signals for business users.")

set_active_dataset_from_sidebar(show_external_uploader=False)
df = load_active_data()

df = load_telco_data()
probability, raw = require_prediction()

# --- Sentiment score mapping (simple + explainable)
sentiment_score = round((1 - probability) * 100)

# --- KPIs
c1, c2, c3 = st.columns(3)
c1.metric("Churn Probability", f"{probability:.2%}")
c2.metric("Sentiment Score", f"{sentiment_score}/100")

if sentiment_score >= 70:
    c3.success("😊 Positive")
    sentiment_label = "Positive"
elif sentiment_score >= 40:
    c3.warning("😐 Neutral")
    sentiment_label = "Neutral"
else:
    c3.error("😡 Negative")
    sentiment_label = "Negative"

st.markdown("---")

# --- Gauge chart (different chart type)
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=sentiment_score,
    number={"suffix": "/100"},
    title={"text": "Sentiment Gauge"},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": "#2563EB"},
        "steps": [
            {"range": [0, 40], "color": "rgba(239,68,68,0.25)"},
            {"range": [40, 70], "color": "rgba(245,158,11,0.25)"},
            {"range": [70, 100], "color": "rgba(34,197,94,0.25)"},
        ],
        "threshold": {"line": {"color": "rgba(15,23,42,0.6)", "width": 2}, "thickness": 0.75, "value": sentiment_score}
    }
))
fig_gauge.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20), paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_gauge, use_container_width=True)

# --- Benchmark band: your risk vs overall churn rate
overall_churn = (df["Churn"] == "Yes").mean()

fig_band = go.Figure()
fig_band.add_trace(go.Bar(
    x=["Your Customer", "Dataset Average"],
    y=[probability, overall_churn],
    text=[f"{probability:.1%}", f"{overall_churn:.1%}"],
    textposition="outside",
))
fig_band.update_layout(
    title="Risk Benchmark (Your Customer vs Dataset Average)",
    height=320,
    yaxis=dict(tickformat=".0%"),
    margin=dict(l=20, r=20, t=60, b=20),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_band, use_container_width=True)

st.markdown("---")

# --- Segment-based sentiment: compare with similar segment
# Create a simple segment based on tenure + charges + contract
seg = df.copy()
seg["tenure_band"] = pd.cut(seg["tenure"], bins=[-1, 12, 24, 72], labels=["0-12", "12-24", "24+"])
seg["charges_band"] = pd.cut(seg["MonthlyCharges"], bins=[-1, 50, 80, 999], labels=["Low", "Mid", "High"])

cust_tenure_band = "0-12" if raw["tenure"] < 12 else ("12-24" if raw["tenure"] < 24 else "24+")
cust_charges_band = "High" if raw["monthly_charges"] > 80 else ("Mid" if raw["monthly_charges"] > 50 else "Low")
cust_contract = raw["contract"]

seg_group = seg[
    (seg["tenure_band"] == cust_tenure_band) &
    (seg["charges_band"] == cust_charges_band) &
    (seg["Contract"] == cust_contract)
]

if len(seg_group) < 30:
    seg_group = seg.copy()

seg_churn = (seg_group["Churn"] == "Yes").mean()

fig_seg = px.bar(
    x=["Similar Segment Churn Rate"],
    y=[seg_churn],
    text=[f"{seg_churn:.1%}"],
    title="Similar Segment Churn Rate (Tenure + Charges + Contract)",
)
fig_seg.update_layout(
    height=300,
    yaxis=dict(tickformat=".0%"),
    margin=dict(l=20, r=20, t=60, b=20),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_seg, use_container_width=True)

st.markdown("---")

# --- Suggested actions + message templates (distinction-level)
st.subheader("🧾 Suggested Messaging (for Retention Team)")

if sentiment_label == "Negative":
    msg = (
        "Hi! We noticed you may not be getting the best value from your plan. "
        "We can offer a personalized discount or upgrade + free support to improve your experience."
    )
elif sentiment_label == "Neutral":
    msg = (
        "Hi! Thanks for being with us. We have some new bundles that can reduce your monthly bill "
        "and improve service quality. Want to explore options?"
    )
else:
    msg = (
        "Hi! Thanks for being a valued customer. If you ever need help, our support team is available. "
        "We also have loyalty rewards you can claim."
    )

st.info(msg)
st.caption("This template is generated based on churn probability → sentiment mapping.")