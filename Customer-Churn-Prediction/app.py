import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("📞 Customer Churn Prediction")

# Inputs
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", [0, 1])

partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])

tenure = st.number_input("Tenure (Months)", 0, 72, 12)

phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=float(monthly_charges * max(tenure, 1))
)

if st.button("Predict Churn"):

    data = pd.DataFrame(
        [[0] * len(feature_names)],
        columns=feature_names
    )

    # Numeric
    data["SeniorCitizen"] = senior
    data["tenure"] = tenure
    data["MonthlyCharges"] = monthly_charges
    data["TotalCharges"] = total_charges

    # Binary
    data["gender_Male"] = (gender == "Male")
    data["Partner_Yes"] = (partner == "Yes")
    data["Dependents_Yes"] = (dependents == "Yes")
    data["PhoneService_Yes"] = (phone_service == "Yes")
    data["PaperlessBilling_Yes"] = (paperless == "Yes")

    # Multiple Lines
    data["MultipleLines_No phone service"] = (
        multiple_lines == "No phone service"
    )
    data["MultipleLines_Yes"] = (
        multiple_lines == "Yes"
    )

    # Internet Service
    data["InternetService_Fiber optic"] = (
        internet_service == "Fiber optic"
    )
    data["InternetService_No"] = (
        internet_service == "No"
    )

    # Online Security
    data["OnlineSecurity_No internet service"] = (
        online_security == "No internet service"
    )
    data["OnlineSecurity_Yes"] = (
        online_security == "Yes"
    )

    # Online Backup
    data["OnlineBackup_No internet service"] = (
        online_backup == "No internet service"
    )
    data["OnlineBackup_Yes"] = (
        online_backup == "Yes"
    )

    # Device Protection
    data["DeviceProtection_No internet service"] = (
        device_protection == "No internet service"
    )
    data["DeviceProtection_Yes"] = (
        device_protection == "Yes"
    )

    # Tech Support
    data["TechSupport_No internet service"] = (
        tech_support == "No internet service"
    )
    data["TechSupport_Yes"] = (
        tech_support == "Yes"
    )

    # Streaming TV
    data["StreamingTV_No internet service"] = (
        streaming_tv == "No internet service"
    )
    data["StreamingTV_Yes"] = (
        streaming_tv == "Yes"
    )

    # Streaming Movies
    data["StreamingMovies_No internet service"] = (
        streaming_movies == "No internet service"
    )
    data["StreamingMovies_Yes"] = (
        streaming_movies == "Yes"
    )

    # Contract
    data["Contract_One year"] = (
        contract == "One year"
    )
    data["Contract_Two year"] = (
        contract == "Two year"
    )

    # Payment Method
    data["PaymentMethod_Credit card (automatic)"] = (
        payment_method == "Credit card (automatic)"
    )

    data["PaymentMethod_Electronic check"] = (
        payment_method == "Electronic check"
    )

    data["PaymentMethod_Mailed check"] = (
        payment_method == "Mailed check"
    )

    # Scale
    scaled_data = scaler.transform(data)

    # Predict
    probability = model.predict_proba(
        scaled_data
    )[0][1]

    prediction = model.predict(
        scaled_data
    )[0]

    st.subheader("Prediction Result")

    st.write(
        f"Churn Probability: {probability:.2%}"
    )

    if probability >= 0.70:
        st.error("🔴 High Churn Risk")

    elif probability >= 0.40:
        st.warning("🟡 Medium Churn Risk")

    else:
        st.success("🟢 Low Churn Risk")

    st.write(
        f"Predicted Class: {'Churn' if prediction == 1 else 'No Churn'}"
    )