import streamlit as st
from utils.data_handler import load_data, preprocess_data
from utils.visualization import show_data_overview, show_feature_importance
from utils.model import train_model, evaluate_model, predict_single, predict_batch
import pandas as pd

st.set_page_config(page_title="Advanced Heart Disease Prediction", layout="wide")
st.title("💓 Heart Disease Detection App")

# Tabs
tabs = st.tabs(["📁 Data Overview", "📊 Feature Selection", "🧠 Model Training", "🔍 Live Prediction", "📂 Batch Prediction"])

# Load dataset
data = load_data()

with tabs[0]:
    st.header("📁 Data Overview")
    show_data_overview(data)

with tabs[1]:
    st.header("📊 Feature Selection")
    X, y, selected_features = preprocess_data(data)

with tabs[2]:
    st.header("🧠 Model Training and Evaluation")
    model, metrics = train_model(X, y)
    evaluate_model(model, X, y, metrics)
    show_feature_importance(model, X)

with tabs[3]:
    st.header("🔍 Predict Single Patient")
    user_input = {}
    for feature in selected_features:
        user_input[feature] = st.number_input(f"{feature}", step=1.0)
    if st.button("Predict"):
        prediction = predict_single(model, user_input)
        st.success(f"Prediction: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")

with tabs[4]:
    st.header("📂 Predict from Uploaded CSV")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        predictions = predict_batch(model, df[selected_features])
        df["Prediction"] = predictions
        st.write(df)
        csv = df.to_csv(index=False).encode()
        st.download_button("Download Results", csv, "predictions.csv", "text/csv")
