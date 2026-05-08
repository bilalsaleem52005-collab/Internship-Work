# =========================================================
# TASK 5 - PERSONAL LOAN ACCEPTANCE PREDICTION DASHBOARD
# KAGGLE DATASET VERSION (deposit column)
# =========================================================

# =========================
# IMPORT LIBRARIES
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Loan Acceptance Dashboard",
    page_icon="💳",
    layout="wide"
)

# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #eef7ff;
}

h1 {
    color: #1565c0;
    font-weight: bold;
}

h2, h3 {
    color: #0d47a1;
}

.stMetric {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    border: 2px solid #bbdefb;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.08);
}

div.stButton > button:first-child {
    background-color: #ff4d6d;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

div.stButton > button:first-child:hover {
    background-color: #e63950;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.title("💳 Personal Loan Acceptance Prediction Dashboard")

st.markdown("""
### Logistic Regression Analysis for Bank Marketing Dataset
""")

# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    # CHANGE PATH IF NEEDED
    df = pd.read_csv("../dataset/bank.csv")

    return df

df = load_data()

# =========================================================
# SHOW COLUMN NAMES
# =========================================================

# st.write(df.columns)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Overview",
        "📊 Data Exploration",
        "📈 Visual Insights",
        "🤖 Model Performance",
        "🧠 Prediction System",
        "💡 Business Insights"
    ]
)

# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "🏠 Overview":

    st.header("📂 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    col4.metric(
        "Accepted Deposits",
        int((df['deposit'] == 'yes').sum())
    )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(df.head(15), use_container_width=True)

    st.markdown("---")

    st.subheader("Statistical Summary")

    st.dataframe(df.describe(), use_container_width=True)

# =========================================================
# DATA EXPLORATION
# =========================================================

elif page == "📊 Data Exploration":

    st.header("📊 Basic Data Exploration")

    option = st.selectbox(
        "Select Analysis",
        [
            "Age Distribution",
            "Job Categories",
            "Marital Status",
            "Education Level"
        ]
    )

    # AGE DISTRIBUTION
    if option == "Age Distribution":

        fig = px.histogram(
            df,
            x='age',
            nbins=30,
            color_discrete_sequence=['#4cc9f0']
        )

        fig.update_layout(
            title="Age Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    # JOB CATEGORY
    elif option == "Job Categories":

        job_counts = df['job'].value_counts().reset_index()

        fig = px.bar(
            job_counts,
            x='job',
            y='count',
            color='count',
            color_continuous_scale='Blues'
        )

        fig.update_layout(
            title="Job Categories"
        )

        st.plotly_chart(fig, use_container_width=True)

    # MARITAL STATUS
    elif option == "Marital Status":

        marital = df['marital'].value_counts().reset_index()

        fig = px.pie(
            marital,
            names='marital',
            values='count',
            color_discrete_sequence=[
                '#ff4d6d',
                '#4cc9f0',
                '#90be6d'
            ]
        )

        st.plotly_chart(fig, use_container_width=True)

    # EDUCATION
    elif option == "Education Level":

        edu = df['education'].value_counts().reset_index()

        fig = px.bar(
            edu,
            x='education',
            y='count',
            color='count',
            color_continuous_scale='Viridis'
        )

        fig.update_layout(
            title="Education Levels"
        )

        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# VISUAL INSIGHTS
# =========================================================

elif page == "📈 Visual Insights":

    st.header("📈 Customer Insights")

    chart = st.selectbox(
        "Select Insight",
        [
            "Job vs Deposit Acceptance",
            "Marital Status vs Acceptance",
            "Age vs Acceptance",
            "Education vs Acceptance"
        ]
    )

    # JOB VS ACCEPTANCE
    if chart == "Job vs Deposit Acceptance":

        fig = px.histogram(
            df,
            x='job',
            color='deposit',
            barmode='group',
            color_discrete_sequence=[
                '#ff4d6d',
                '#2ecc71'
            ]
        )

        fig.update_layout(
            title="Job Type vs Deposit Acceptance"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.success(
            "Certain professions are more likely to accept deposit offers."
        )

    # MARITAL STATUS
    elif chart == "Marital Status vs Acceptance":

        fig = px.histogram(
            df,
            x='marital',
            color='deposit',
            barmode='group',
            color_discrete_sequence=[
                '#ff4d6d',
                '#2ecc71'
            ]
        )

        fig.update_layout(
            title="Marital Status vs Deposit Acceptance"
        )

        st.plotly_chart(fig, use_container_width=True)

    # AGE VS ACCEPTANCE
    elif chart == "Age vs Acceptance":

        fig = px.box(
            df,
            x='deposit',
            y='age',
            color='deposit',
            color_discrete_sequence=[
                '#ff4d6d',
                '#2ecc71'
            ]
        )

        fig.update_layout(
            title="Age vs Deposit Acceptance"
        )

        st.plotly_chart(fig, use_container_width=True)

    # EDUCATION
    elif chart == "Education vs Acceptance":

        fig = px.histogram(
            df,
            x='education',
            color='deposit',
            barmode='group',
            color_discrete_sequence=[
                '#ff4d6d',
                '#2ecc71'
            ]
        )

        fig.update_layout(
            title="Education vs Deposit Acceptance"
        )

        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "🤖 Model Performance":

    st.header("🤖 Logistic Regression Model")

    model_df = df.copy()

    # ENCODE CATEGORICAL VARIABLES
    encoders = {}

    for column in model_df.columns:

        if model_df[column].dtype == 'object':

            le = LabelEncoder()

            model_df[column] = le.fit_transform(
                model_df[column]
            )

            encoders[column] = le

    # FEATURES & TARGET
    X = model_df.drop('deposit', axis=1)

    y = model_df['deposit']

    # SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # TRAIN MODEL
    model = LogisticRegression(max_iter=5000)

    model.fit(X_train, y_train)

    # PREDICTIONS
    predictions = model.predict(X_test)

    # METRICS
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Model Accuracy",
        f"{accuracy:.2%}"
    )

    col2.metric(
        "Training Samples",
        len(X_train)
    )

    st.markdown("---")

    # CONFUSION MATRIX
    st.subheader("📌 Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        predictions
    )

    fig, ax = plt.subplots(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    st.pyplot(fig)

    st.markdown("---")

    # FEATURE IMPORTANCE
    st.subheader("📈 Feature Importance")

    importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_[0]
    })

    importance = importance.sort_values(
        by='Coefficient',
        ascending=False
    )

    fig2 = px.bar(
        importance,
        x='Feature',
        y='Coefficient',
        color='Coefficient',
        color_continuous_scale='RdYlGn'
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.subheader("📋 Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df,
        use_container_width=True
    )

# =========================================================
# PREDICTION SYSTEM
# =========================================================

elif page == "🧠 Prediction System":

    st.header("🧠 Predict Deposit Acceptance")

    pred_df = df.copy()

    encoders = {}

    # ENCODE
    for column in pred_df.columns:

        if pred_df[column].dtype == 'object':

            le = LabelEncoder()

            pred_df[column] = le.fit_transform(
                pred_df[column]
            )

            encoders[column] = le

    # FEATURES & TARGET
    X = pred_df.drop('deposit', axis=1)

    y = pred_df['deposit']

    # TRAIN MODEL
    model = LogisticRegression(max_iter=5000)

    model.fit(X, y)

    # INPUT SECTION
    col1, col2 = st.columns(2)

    with col1:

        age = st.slider(
            "Age",
            18,
            90,
            30
        )

        job = st.selectbox(
            "Job",
            df['job'].unique()
        )

        marital = st.selectbox(
            "Marital Status",
            df['marital'].unique()
        )

        education = st.selectbox(
            "Education",
            df['education'].unique()
        )

    with col2:

        balance = st.slider(
            "Balance",
            0,
            100000,
            2000
        )

        housing = st.selectbox(
            "Housing Loan",
            df['housing'].unique()
        )

        loan = st.selectbox(
            "Personal Loan",
            df['loan'].unique()
        )

        contact = st.selectbox(
            "Contact Type",
            df['contact'].unique()
        )

    # CREATE INPUT DATA
    input_dict = {

        'age': age,

        'job':
        encoders['job'].transform([job])[0],

        'marital':
        encoders['marital'].transform([marital])[0],

        'education':
        encoders['education'].transform([education])[0],

        'default': 0,

        'balance': balance,

        'housing':
        encoders['housing'].transform([housing])[0],

        'loan':
        encoders['loan'].transform([loan])[0],

        'contact':
        encoders['contact'].transform([contact])[0],

        'day': 5,

        'month':
        encoders['month'].transform(['may'])[0],

        'duration': 200,

        'campaign': 1,

        'pdays': -1,

        'previous': 0,

        'poutcome':
        encoders['poutcome'].transform(['unknown'])[0]

    }

    input_df = pd.DataFrame([input_dict])

    # PREDICTION BUTTON
    if st.button("Predict Acceptance"):

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(
            input_df
        )[0][1]

        if prediction == 1:

            st.success(
                f"""
                ✅ Customer is LIKELY to accept the deposit offer.

                Probability: {probability:.2%}
                """
            )

        else:

            st.error(
                f"""
                ❌ Customer is NOT likely to accept the deposit offer.

                Probability: {probability:.2%}
                """
            )

# =========================================================
# BUSINESS INSIGHTS
# =========================================================

elif page == "💡 Business Insights":

    st.header("💡 Business Insights")

    st.markdown("""

## 📌 Key Findings

### ✅ Customers More Likely to Accept Deposits
- Customers with higher balances
- Customers without existing loans
- Certain job categories
- Middle-aged customers

---

## 📈 Marketing Recommendations

- Target financially stable customers
- Focus campaigns on positive-balance accounts
- Use personalized marketing strategies
- Prioritize customer groups with high acceptance probability

---

## 🤖 Machine Learning Benefits

- Reduces marketing costs
- Improves targeting efficiency
- Helps banks make data-driven decisions
- Predicts customer behavior effectively

---

## 🛠 Technologies Used

- Python
- Streamlit
- Plotly
- Scikit-learn
- Logistic Regression

""")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
### 🚀 Developed for Data Science & Analytics Internship Task 5
""")