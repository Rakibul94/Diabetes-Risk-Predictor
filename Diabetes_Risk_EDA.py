
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder




st.title("Exploratory Data Analysis on Diabetes Risk")
st.sidebar.markdown("EDA")

df = pd.read_csv("diabetes.csv")


st.subheader("Statistical Summary")
st.table(df.describe())




# Create the plot
st.subheader("Correlation Matrix")

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)


# Show it in Streamlit
st.pyplot(fig)

st.subheader("📊 Feature Distributions")

# Loop through columns (excluding last one, e.g. 'Outcome')
for col in df.columns:
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f'Distribution of {col}')
    st.pyplot(fig)

#Box Plots
st.subheader("📦 Box Plots")

for col in df.columns[:-1]:
    fig, ax = plt.subplots()
    sns.boxplot(x='Outcome', y=col, data=df, ax=ax)
    ax.set_title(f'{col} by Diabetes Outcome')
    st.pyplot(fig)


lr_df = pd.read_csv("logistic_coefficients_diabetes.csv")
rf_df = pd.read_csv("randomforest_featureimportance_diabetes.csv")

st.subheader("Logistic Regression Model Coefficients")
st.write(lr_df)

st.subheader("Feature Importance using Random Forest")

st.write(rf_df)