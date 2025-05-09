import streamlit as st

# Define the pages
main_page = st.Page("Diabetes_Risk_Predictor_app.py", title="Diabetes Risk Predictor")
page_2 = st.Page("Diabetes_Risk_EDA.py", title="EDA on Diabetes Risk")


# Set up navigation
pg = st.navigation([main_page, page_2])

# Run the selected page
pg.run()