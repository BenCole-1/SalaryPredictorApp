import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Load the regression model and feature columns
try:
    with open("bus458_finalmodel.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_cols.json") as f:
        model_columns = json.load(f)
except FileNotFoundError:
    st.error("❌ Required model files not found. Please upload 'bus458_finalmodel.pkl' and 'feature_cols.json'.")
    st.stop()

# --- Interface ---

# Title
st.markdown(
    "<h1 style='text-align: center; background-color: #002b36; padding: 15px; color: #ffffff; border-radius: 8px;'>💼 <b>Data Scientist Salary Predictor</b></h1>",
    unsafe_allow_html=True
)

# Header
st.markdown("<h3 style='color:#007acc;'>📋 Enter Your Information</h3>", unsafe_allow_html=True)

# Inputs
years_coding = st.slider("👨‍💻 Years of Coding Experience", min_value=0, max_value=50, value=5)
years_ml = st.slider("🧠 Years of Machine Learning Experience", min_value=0, max_value=2, value=2)
money_spent = st.number_input("💸 Money Spent on ML/Cloud in Last 5 Years ($USD)", min_value=0, max_value=100000, value=1000)

country = st.selectbox("🌍 Country You Reside In", [
    'France', 'Other', 'Australia', 'United States of America',
    'Italy', 'Brazil', 'Argentina', 'Japan', 'Canada', 'India',
    'Colombia', 'Pakistan', 'Mexico', 'Turkey', 'Nigeria', 'Tunisia',
    'Philippines', 'South Korea', 'Peru', 'Iran, Islamic Republic of...', 'Russia',
    'Bangladesh', 'Israel', 'Kenya'
])

job_title = st.selectbox("👔 Current Role", [
    'Data Scientist', 'Software Engineer', 'Research Scientist', 'Developer Advocate',
    'Data Analyst (Business, Marketing, Financial, Quantitative, etc)', 'Data Engineer', 'Other',
    'Machine Learning/ MLops Engineer', 'Engineer (non-software)', 'Teacher / professor',
    'Manager (Program, Project, Operations, Executive-level, etc)', 'Statistician',
    'Data Administrator', 'Data Architect'
])

industry = st.selectbox("🏢 Industry of Current Employer", [
    'Online Service/Internet-based Services', 'Insurance/Risk Assessment', 'Government/Public Service',
    'Computers/Technology', 'Accounting/Finance', 'Academics/Education', 'Non-profit/Service',
    'Other', 'Medical/Pharmaceutical', 'Marketing/CRM', 'Manufacturing/Fabrication',
    'Energy/Mining', 'Broadcasting/Communications', 'Retail/Sales', 'Shipping/Transportation'
])

ml_incorporated = st.selectbox("🧠 Does Your Employer Use ML Methods?", [
    'We recently started using ML methods (i.e., models in production for less than 2 years)',
    'We have well established ML methods (i.e., models in production for more than 2 years)',
    'We are exploring ML methods (and may one day put a model into production)',
    'I do not know',
    'We use ML methods for generating insights (but do not put working models into production)',
    'No (we do not use ML methods)'
])

# --- Data Processing ---
# Build input DataFrame (correct columns)
input_data = pd.DataFrame({
    "Q11": [years_coding],
    "Q16": [years_ml],
    "Q30": [money_spent],
    "Q4": [country],
    "Q23": [job_title],
    "Q24": [industry],
    "Q27": [ml_incorporated]
})

# Mapping based on training
experience_map = {
    'I have never written code': 0, '< 1 years': 0.5, '1-2 years': 1.5, '3-5 years': 4,
    '5-10 years': 7.5, '10-20 years': 15, '20+ years': 25
}
ml_exp_map = {
    'I do not use machine learning methods': 0, 'Under 1 year': 0.5, '1-2 years': 1.5,
    '2-3 years': 2.5, '3-4 years': 3.5, '4-5 years': 4.5, '5-10 years': 7.5,
    '10-20 years': 15, '20 or more years': 25
}
spend_map = {
    '$0 ($USD)': 0, '$1-$99': 50, '$100-$999': 550, '$1000-$9,999': 5000,
    '$10,000-$99,999': 50000, '$100,000 or more ($USD)': 100000
}
ml_maturity_map = {
    'No (we do not use ML methods)': 0,
    'We are exploring ML methods (and may one day put a model into production)': 1,
    'We use ML methods for generating insights (but do not put working models into production)': 2,
    'We recently started using ML methods (i.e., models in production for less than 2 years)': 3,
    'We have well established ML methods (i.e., models in production for more than 2 years)': 4
}

# Apply mappings
input_data["experience_years"] = input_data["Q11"].map(experience_map)
input_data["ml_experience_years"] = input_data["Q16"].map(ml_exp_map)
input_data["cloud_spend"] = input_data["Q30"].map(spend_map)
input_data["ml_maturity"] = input_data["Q27"].map(ml_maturity_map)

# Rename columns for consistency
input_data["role"] = input_data["Q23"]
input_data["country"] = input_data["Q4"]
input_data["industry"] = input_data["Q24"]

# Select correct columns for model
input_data = input_data[["experience_years", "ml_experience_years", "cloud_spend", "ml_maturity", "role", "country", "industry"]]

# One-hot encode categorical features
input_data_encoded = pd.get_dummies(input_data, columns=["role", "country", "industry"])

# Fill missing columns
for col in model_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# Reorder columns
input_data_encoded = input_data_encoded[model_columns]

# --- Prediction ---
if st.button("💰 Predict Salary"):
    with st.spinner('Predicting salary...'):
        prediction_log = model.predict(input_data_encoded)[0]
        predicted_salary = np.expm1(prediction_log)  # revert log1p
    st.success(f"🎯 Estimated Yearly Compensation: **${predicted_salary:,.2f}**")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; font-size: 13px;'>✨ <i>Application Created By Ben Cole</i></div>", unsafe_allow_html=True)
