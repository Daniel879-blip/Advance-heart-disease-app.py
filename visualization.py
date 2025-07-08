import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def show_data_overview(df):
    st.write("Dataset Preview", df.head())
    st.write("Dataset Shape:", df.shape)
    st.write("Missing Values:", df.isnull().sum())
    if st.checkbox("Show Correlation Heatmap"):
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

def show_feature_importance(model, X):
    import pandas as pd
    st.subheader("Feature Importances")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=False)
    st.bar_chart(feat_df.set_index("Feature"))
