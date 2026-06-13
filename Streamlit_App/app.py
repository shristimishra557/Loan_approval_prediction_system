import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open("../Model/best_model.pkl", "rb"))
scaler = pickle.load(open("../Model/scaler.pkl", "rb"))

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦"
)

st.title("🏦 Loan Approval Prediction System")

st.write("Enter applicant details below:")

# User Inputs
gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

married = st.selectbox(
    "Married",
    ["No", "Yes"]
)

dependents = st.selectbox(
    "Dependents",
    [0, 1, 2, 3]
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["No", "Yes"]
)

applicant_income = st.number_input(
    "Applicant Income",
    min_value=0
)

coapplicant_income = st.number_input(
    "Coapplicant Income",
    min_value=0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0
)

loan_term = st.number_input(
    "Loan Amount Term",
    value=360
)

credit_history = st.selectbox(
    "Credit History",
    [0, 1]
)

property_area = st.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"]
)

# Encoding
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 0 if education == "Graduate" else 1
self_employed = 1 if self_employed == "Yes" else 0

if property_area == "Rural":
    property_area = 0
elif property_area == "Semiurban":
    property_area = 1
else:
    property_area = 2

# Feature Engineering
total_income = applicant_income + coapplicant_income

income_loan_ratio = (
    total_income / (loan_amount + 1)
)

# Input Data
input_data = pd.DataFrame([[
    gender,
    married,
    dependents,
    education,
    self_employed,
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_term,
    credit_history,
    property_area,
    total_income,
    income_loan_ratio
]], columns=[
    'Gender',
    'Married',
    'Dependents',
    'Education',
    'Self_Employed',
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Loan_Amount_Term',
    'Credit_History',
    'Property_Area',
    'TotalIncome',
    'Income_Loan_Ratio'
])

# Scale
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Loan Status"):

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")