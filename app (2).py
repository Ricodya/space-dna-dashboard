import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
import plotly.express as px

st.set_page_config(page_title="Space DNA Mutation Predictor", layout="wide")

# ------------------ NASA HEADER ------------------
st.markdown("""
    <h1 style='text-align:center; color:#00BFFF;'>🛰 SPACE DNA MUTATION PREDICTION DASHBOARD</h1>
    <p style='text-align:center; color:#FFFFFF;'>Radiation | Microgravity | Cosmic Rays | Temperature</p>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.title("⚙ Settings")
option = st.sidebar.selectbox(
    "Select Model",
    ("Support Vector Machine (SVM)",)
)

st.sidebar.markdown("---")
st.sidebar.info("Upload dataset with features:\n- radiation\n- gravity\n- cosmic_rays\n- temperature\n- mutation_risk")

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("📥 Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✔ File Uploaded Successfully")
    
    st.subheader("📊 Dataset Preview")
    st.dataframe(df)

    # ------------------ PREPROCESSING ------------------
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ------------------ MODEL TRAINING ------------------
    st.subheader("🚀 Model Training Status")

    model = SVC(probability=True)
    model.fit(X_scaled, y)

    preds = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)

    acc = accuracy_score(y, preds)

    st.success(f"Model Trained Successfully with Accuracy: **{acc*100:.2f}%**")

    # ---------------- CONFUSION MATRIX ------------------
    st.subheader("🔍 Confusion Matrix")
    cm = confusion_matrix(y, preds)
    st.write(cm)

    # ------------------ BAR GRAPH ------------------
    st.subheader("📈 Mutation Risk Prediction Graph")

    graph_df = pd.DataFrame({
        "Actual": y,
        "Predicted": preds
    })

    fig = px.bar(graph_df, barmode="group", title="Actual vs Predicted Mutation Risk")
    st.plotly_chart(fig, use_container_width=True)

    # ------------------ SINGLE PREDICTION ------------------
    st.subheader("🔮 Predict Mutation Risk for New Values")

    radiation = st.number_input("Radiation", 0.0)
    gravity = st.number_input("Microgravity", 0.0)
    cosmic = st.number_input("Cosmic Rays", 0.0)
    temp = st.number_input("Temperature", 0.0)

    if st.button("Predict"):
        new_data = scaler.transform([[radiation, gravity, cosmic, temp]])
        pred_new = model.predict(new_data)[0]
        prob = model.predict_proba(new_data)[0][1] * 100

        st.success(f"🧬 Predicted Mutation Risk: **{pred_new}**")
        st.info(f"Probability: {prob:.2f}%")

else:
    st.warning("⚠ Please upload your dataset to continue.")
