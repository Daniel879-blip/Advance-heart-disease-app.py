
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set page configuration
st.set_page_config(page_title="Heart Disease Prediction App", layout="wide")

# Title
st.title("❤️ Heart Disease Prediction App")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")
    return df

# Train model
def train_model(X, y):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
    return model

# Visualize dataset
def visualize_data(df):
    st.subheader("📊 Data Visualization")
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("### Target Distribution")
        fig2, ax2 = plt.subplots()
        sns.countplot(data=df, x="target", palette="Set2", ax=ax2)
        st.pyplot(fig2)

# Main App
def main():
    menu = ["1️⃣ Overview", "2️⃣ Visualize", "3️⃣ Predict"]
    choice = st.sidebar.radio("Navigation", menu)

    df = load_data()

    if choice == "1️⃣ Overview":
        st.subheader("📝 Dataset Overview")
        st.dataframe(df.head())
        st.write("Shape of dataset:", df.shape)
        st.write("Missing values:", df.isnull().sum().sum())

    elif choice == "2️⃣ Visualize":
        visualize_data(df)

    elif choice == "3️⃣ Predict":
        st.subheader("🤖 Heart Disease Prediction")

        # User input
        def user_input_features():
            age = st.slider("Age", 29, 77, 54)
            sex = st.selectbox("Sex", [0, 1])
            cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
            trestbps = st.slider("Resting Blood Pressure", 94, 200, 130)
            chol = st.slider("Serum Cholesterol", 126, 564, 246)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
            restecg = st.selectbox("Resting ECG", [0, 1, 2])
            thalach = st.slider("Max Heart Rate Achieved", 71, 202, 150)
            exang = st.selectbox("Exercise Induced Angina", [0, 1])
            oldpeak = st.slider("ST Depression", 0.0, 6.2, 1.0)
            slope = st.selectbox("Slope of the ST Segment", [0, 1, 2])
            ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
            thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

            data = {
                "age": age,
                "sex": sex,
                "cp": cp,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs,
                "restecg": restecg,
                "thalach": thalach,
                "exang": exang,
                "oldpeak": oldpeak,
                "slope": slope,
                "ca": ca,
                "thal": thal,
            }

            return pd.DataFrame([data])

        input_df = user_input_features()

        if st.button("Predict"):
            if "target" not in df.columns:
                st.error("❌ Dataset missing 'target' column.")
                return

            X = df.drop("target", axis=1)
            y = df["target"]

            if len(y.unique()) < 2:
                st.error("❌ The 'target' column must contain at least two classes.")
                return

            model = train_model(X, y)
            prediction = model.predict(input_df)

            if prediction[0] == 1:
                st.success("✅ The model predicts a high risk of heart disease.")
            else:
                st.info("✅ The model predicts a low risk of heart disease.")

if __name__ == "__main__":
    main()
