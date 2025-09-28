import streamlit as st
import pandas as pd
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Telco Churn Prediction")
st.markdown("Fill in the customer details to predict churn probability.")

# Inputs
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
tenure = st.number_input("Tenure (months)", 0, 100, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 1000.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, tenure*monthly_charges)

contract = st.selectbox("Contract Type", ["Month-to-Month", "One year", "Two year"])
dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
partner = st.selectbox("Has Partner?", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", ["Credit card (automatic)", "Electronic check", "Mailed check"])
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
gender = st.selectbox("Gender", ["Female", "Male"])

# Mapping inputs to model features
feature_dict = {
    "SeniorCitizen": 1 if senior=="Yes" else 0,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract_One year": 1 if contract=="One year" else 0,
    "Contract_Two year": 1 if contract=="Two year" else 0,
    "Dependents_1.0": 1 if dependents=="Yes" else 0,
    "DeviceProtection_No internet service": 1 if device_protection=="No internet service" else 0,
    "DeviceProtection_Yes": 1 if device_protection=="Yes" else 0,
    "InternetService_Fiber optic": 1 if internet_service=="Fiber optic" else 0,
    "InternetService_No": 1 if internet_service=="No" else 0,
    "MultipleLines_No phone service": 1 if multiple_lines=="No phone service" else 0,
    "MultipleLines_Yes": 1 if multiple_lines=="Yes" else 0,
    "OnlineBackup_No internet service": 1 if online_backup=="No internet service" else 0,
    "OnlineBackup_Yes": 1 if online_backup=="Yes" else 0,
    "OnlineSecurity_No internet service": 1 if online_security=="No internet service" else 0,
    "OnlineSecurity_Yes": 1 if online_security=="Yes" else 0,
    "PaperlessBilling_1.0": 1 if paperless_billing=="Yes" else 0,
    "Partner_1.0": 1 if partner=="Yes" else 0,
    "PaymentMethod_Credit card (automatic)": 1 if payment_method=="Credit card (automatic)" else 0,
    "PaymentMethod_Electronic check": 1 if payment_method=="Electronic check" else 0,
    "PaymentMethod_Mailed check": 1 if payment_method=="Mailed check" else 0,
    "PhoneService_1.0": 1 if phone_service=="Yes" else 0,
    "StreamingMovies_No internet service": 1 if streaming_movies=="No internet service" else 0,
    "StreamingMovies_Yes": 1 if streaming_movies=="Yes" else 0,
    "StreamingTV_No internet service": 1 if streaming_tv=="No internet service" else 0,
    "StreamingTV_Yes": 1 if streaming_tv=="Yes" else 0,
    "TechSupport_No internet service": 1 if tech_support=="No internet service" else 0,
    "TechSupport_Yes": 1 if tech_support=="Yes" else 0,
    "gender_Male": 1 if gender=="Male" else 0
}

# Ensure correct column order
model_features = model.get_booster().feature_names  # xgboost trained feature names
X_new = pd.DataFrame([feature_dict], columns=model_features)  # force same column order

# Prediction
if st.button("Predict Churn"):
    prediction = model.predict(X_new)[0]
    probability = model.predict_proba(X_new)[0][1]
    
    st.subheader(f"Prediction: {'Churn' if prediction==1 else 'No Churn'}")
    st.subheader(f"Probability of Churn: {probability:.2f}")
