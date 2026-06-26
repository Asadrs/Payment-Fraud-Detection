import plotly.express as px
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("models/fraud_model.pkl")

# Page settings
st.set_page_config(
    page_title="Payment Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.hero {
    padding: 35px;
    border-radius: 20px;
    background: linear-gradient(135deg, #0f172a, #1e3a8a, #2563eb);
    text-align: center;
    margin-bottom: 30px;
}
.hero h1 {
    color: white;
    font-size: 46px;
    font-weight: 900;
}
.hero p {
    color: #dbeafe;
    font-size: 18px;
}
.card {
    background-color: #111827;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #374151;
}
.card h2 {
    font-size: 38px;
    margin: 0;
}
.card p {
    color: #d1d5db;
    margin: 0;
}
.footer {
    text-align: center;
    color: #9ca3af;
    padding: 25px;
    margin-top: 40px;
}
.alert-box {
    background:#7f1d1d;
    padding:20px;
    border-radius:15px;
    color:white;
    font-size:22px;
    text-align:center;
    margin-bottom:20px;
}
.success-box {
    background:#14532d;
    padding:20px;
    border-radius:15px;
    color:white;
    font-size:22px;
    text-align:center;
    margin-bottom:20px;
}
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero">
    <h1>💳 AI-Powered Payment Fraud Detection</h1>
    <p>Machine Learning Dashboard using Random Forest, SMOTE, and Streamlit</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📌 About Project")
    st.write("This dashboard detects fraudulent credit card transactions using a trained Random Forest model.")
    st.write("Upload a CSV file containing Time, V1-V28, and Amount columns.")

    st.header("🛠 Technologies")
    st.write("Python")
    st.write("Pandas")
    st.write("Scikit-learn")
    st.write("SMOTE")
    st.write("Random Forest")
    st.write("Streamlit")

    st.header("📊 Model Details")
    st.write("Model: Random Forest")
    st.write("Accuracy: 99.95%")
    st.write("ROC-AUC: 97.31%")
    st.write("SMOTE Applied: Yes")

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload Transaction CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    required_columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(data.head(10), use_container_width=True)

    if all(col in data.columns for col in required_columns):
        X = data[required_columns]

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        data["Prediction"] = predictions
        data["Fraud_Probability"] = probabilities
        data["Transaction_Status"] = data["Prediction"].map({
            0: "Legitimate",
            1: "Fraudulent"
        })

        total_count = len(data)
        fraud_count = int(data["Prediction"].sum())
        legit_count = total_count - fraud_count
        fraud_rate = (fraud_count / total_count) * 100

        # Alert Box
        if fraud_count > 0:
            st.markdown(
                f"""
                <div class="alert-box">
                    ⚠️ {fraud_count} Suspicious Transactions Detected
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="success-box">
                    ✅ No Fraudulent Transactions Detected
                </div>
                """,
                unsafe_allow_html=True
            )

        st.progress(fraud_rate / 100)
        st.write(f"Fraud Rate: {fraud_rate:.2f}%")

        # Prediction Summary
        st.subheader("📊 Prediction Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="card">
                <h2 style="color:#38bdf8;">{total_count}</h2>
                <p>Total Transactions</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="card">
                <h2 style="color:#ef4444;">{fraud_count}</h2>
                <p>Fraudulent</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="card">
                <h2 style="color:#22c55e;">{legit_count}</h2>
                <p>Legitimate</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="card">
                <h2 style="color:#facc15;">{fraud_rate:.2f}%</h2>
                <p>Fraud Rate</p>
            </div>
            """, unsafe_allow_html=True)

        # Model Performance
        st.subheader("📌 Model Performance")

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric("Accuracy", "99.95%")

        with m2:
            st.metric("ROC-AUC", "97.31%")

        with m3:
            st.metric("Fraud Precision", "85%")

        with m4:
            st.metric("Fraud Recall", "84%")

        # Evaluation Visuals
        st.subheader("📷 Model Evaluation Visuals")

        col_img1, col_img2 = st.columns(2)

        with col_img1:
            st.image(
                "outputs/confusion_matrix.png",
                caption="Confusion Matrix",
                use_container_width=True
            )

        with col_img2:
            st.image(
                "outputs/roc_curve.png",
                caption="ROC Curve",
                use_container_width=True
            )

                # Interactive Bar Chart
        st.subheader("📈 Fraud vs Legitimate Transactions")

        chart_data = pd.DataFrame({
            "Transaction Type": ["Legitimate", "Fraudulent"],
            "Count": [legit_count, fraud_count]
        })

        fig = px.bar(
            chart_data,
            x="Transaction Type",
            y="Count",
            color="Transaction Type",
            text="Count",
            title="Transaction Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)
        

                # Interactive Pie Chart
        st.subheader("🥧 Fraud Percentage Breakdown")

        fig2 = px.pie(
            chart_data,
            names="Transaction Type",
            values="Count",
            title="Fraud vs Legitimate Percentage"
        )

        st.plotly_chart(fig2, use_container_width=True)

                # Fraud Probability Distribution
        st.subheader("📉 Fraud Probability Distribution")

        fig3 = px.histogram(
            data,
            x="Fraud_Probability",
            nbins=50,
            title="Distribution of Fraud Probability Scores"
        )

        st.plotly_chart(fig3, use_container_width=True)

        # Top Fraud Transactions
        st.subheader("🚨 Top Fraud Transactions")

        fraud_df = data[data["Prediction"] == 1]

        if len(fraud_df) > 0:
            st.dataframe(
                fraud_df.head(20),
                use_container_width=True
            )
        else:
            st.success("No fraudulent transactions found in the uploaded file.")

        # Prediction Results
        st.subheader("🔍 Prediction Results")
        st.dataframe(data.head(100), use_container_width=True)

        # Download Results
        csv = data.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download Prediction Results",
            data=csv,
            file_name="fraud_prediction_results.csv",
            mime="text/csv"
        )

    else:
        st.error("CSV must contain Time, V1-V28, and Amount columns.")

else:
    st.info("Upload a CSV file to begin fraud detection.")

# Footer
st.markdown("""
---
<center>
AI-Powered Payment Fraud Detection Dashboard
<br>
Random Forest • SMOTE • Streamlit
<br>
Developed by Asaduddin R.S.
</center>
""", unsafe_allow_html=True)