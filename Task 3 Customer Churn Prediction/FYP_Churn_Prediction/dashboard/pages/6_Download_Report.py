import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from style import apply_custom_style
from style import theme_selector
theme_selector()
from utils import load_telco_data, require_prediction
from utils import set_active_dataset_from_sidebar, load_active_data

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors


st.set_page_config(page_title="Download Report", page_icon="📄", layout="wide")
apply_custom_style()

st.title("📄 Generate & Download Customer Report")

set_active_dataset_from_sidebar(show_external_uploader=False)
df = load_active_data()

df = load_active_data()
probability, raw = require_prediction()

st.markdown("### Customer Summary")
st.write(raw)
st.metric("Predicted Churn Probability", f"{probability:.2%}")

# ---------- Helpers to create chart images ----------
def save_fig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()

def chart_prob_and_loss(prob, loss, out_path):
    plt.figure(figsize=(6.5, 3.2))
    plt.bar(["Churn Probability", "Expected Monthly Loss"], [prob, loss])
    plt.title("Risk & Financial Impact")
    plt.ylabel("Value")
    save_fig(out_path)

def chart_contract_churn(df, out_path):
    grp = df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean()).sort_values(ascending=False)
    plt.figure(figsize=(6.5, 3.2))
    plt.bar(grp.index, grp.values)
    plt.title("Dataset: Churn Rate by Contract Type")
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=0)
    save_fig(out_path)

def chart_monthly_dist(df, customer_value, out_path):
    plt.figure(figsize=(6.5, 3.2))
    plt.hist(df["MonthlyCharges"], bins=30)
    plt.axvline(customer_value, linewidth=3, linestyle="--")
    plt.title("Monthly Charges Distribution (Customer marked)")
    plt.xlabel("Monthly Charges")
    plt.ylabel("Customers")
    save_fig(out_path)

def build_table(data_rows, col_widths=None):
    t = Table(data_rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#EAF3FA")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#0B1220")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    return t

if st.button("📄 Generate Detailed PDF Report"):
    expected_loss = raw["monthly_charges"] * probability
    overall_churn_rate = (df["Churn"] == "Yes").mean()

    # Temporary image paths (saved in working directory)
    img1 = "chart_risk_loss.png"
    img2 = "chart_contract_churn.png"
    img3 = "chart_monthly_dist.png"

    chart_prob_and_loss(probability, expected_loss, img1)
    chart_contract_churn(df, img2)
    chart_monthly_dist(df, raw["monthly_charges"], img3)

    file_path = "Churn_Report_Detailed.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Customer Churn Risk Report", styles["Title"]))
    elements.append(Paragraph("AI-Based Customer Churn Prediction System (FYP)", styles["Normal"]))
    elements.append(Spacer(1, 14))

    # Executive summary
    elements.append(Paragraph("Executive Summary", styles["Heading2"]))
    summary_text = (
        f"This report summarizes churn risk for the selected customer using an XGBoost model. "
        f"The predicted churn probability is <b>{probability:.2%}</b>. "
        f"The expected monthly revenue at risk (MonthlyCharges × Probability) is <b>${expected_loss:.2f}</b>. "
        f"For context, the dataset average churn rate is <b>{overall_churn_rate:.2%}</b>."
    )
    elements.append(Paragraph(summary_text, styles["BodyText"]))
    elements.append(Spacer(1, 12))

    # KPI table
    elements.append(Paragraph("Key Metrics", styles["Heading2"]))
    kpi_rows = [
        ["Metric", "Value"],
        ["Churn Probability", f"{probability:.2%}"],
        ["Risk Level", "High" if probability >= 0.7 else ("Medium" if probability >= 0.4 else "Low")],
        ["Expected Monthly Revenue at Risk", f"${expected_loss:.2f}"],
        ["Dataset Average Churn Rate", f"{overall_churn_rate:.2%}"],
    ]
    elements.append(build_table(kpi_rows, col_widths=[220, 280]))
    elements.append(Spacer(1, 12))

    # Inputs table
    elements.append(Paragraph("Customer Inputs", styles["Heading2"]))
    input_rows = [["Feature", "Value"]] + [[k, str(v)] for k, v in raw.items()]
    elements.append(build_table(input_rows, col_widths=[220, 280]))
    elements.append(Spacer(1, 12))

    # Charts
    elements.append(Paragraph("Charts & Evidence", styles["Heading2"]))
    elements.append(Paragraph("1) Risk & Financial Impact", styles["BodyText"]))
    elements.append(RLImage(img1, width=480, height=240))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("2) Dataset churn patterns (Contract Type)", styles["BodyText"]))
    elements.append(RLImage(img2, width=480, height=240))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("3) Customer cost position (Monthly Charges distribution)", styles["BodyText"]))
    elements.append(RLImage(img3, width=480, height=240))
    elements.append(Spacer(1, 12))

    # Recommendations
    elements.append(Paragraph("Recommended Retention Actions", styles["Heading2"]))
    recs = []

    if raw["contract"] == "Month-to-month":
        recs.append("Offer a discounted long-term contract to reduce churn risk.")
    if raw["monthly_charges"] > 80:
        recs.append("Provide a loyalty discount or bundle upgrade to improve value perception.")
    if raw["tech_support"] == "No":
        recs.append("Offer free tech support for 3 months to improve experience.")
    if raw["online_security"] == "No":
        recs.append("Bundle online security service to increase perceived service value.")
    if raw["tenure"] < 12:
        recs.append("Target onboarding rewards and early engagement to improve retention.")

    if not recs:
        recs.append("Maintain current service quality and monitor engagement.")

    for r in recs:
        elements.append(Paragraph(f"• {r}", styles["BodyText"]))

    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Conclusion", styles["Heading2"]))
    elements.append(Paragraph(
        "This report provides a decision-support view for retention teams by combining prediction, "
        "financial risk estimation, dataset context, and recommended actions.",
        styles["BodyText"]
    ))

    doc.build(elements)

    # Cleanup chart images (optional)
    for p in [img1, img2, img3]:
        try:
            os.remove(p)
        except:
            pass

    with open(file_path, "rb") as f:
        st.download_button("⬇️ Download Detailed Report PDF", f, file_name="Churn_Report_Detailed.pdf")