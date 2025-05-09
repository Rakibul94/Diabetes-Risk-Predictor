
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

model = joblib.load('diabetes_model.pkl')

model2 = joblib.load('diabetes_model2.pkl')

df = pd.read_csv("diabetes.csv")

print(df.iloc[12])


st.title("Diabetes Risk Predictor")


st.markdown("Please provide patient details:")

pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

# When user clicks "Predict"
if st.button("Predict Diabetes Risk"):
    # Create input array for prediction
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, diabetes_pedigree, age]])
    
    # Get prediction
    prediction = model2.predict(input_data)[0]

    # Display result
    if prediction == 1:
        st.error("High risk of Diabetes.")
    else:
        st.success("Low risk of Diabetes.")
