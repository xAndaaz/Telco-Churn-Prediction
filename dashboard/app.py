import os
import streamlit as st
import pandas as pd
import requests
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- 1. HEADER ---
st.title("Churn Prediction and Retention Dashboard")
st.markdown("This dashboard provides insights into customer churn predictions and helps generate retention strategies.")

# --- 2. RETENTION CANDIDATES OVERVIEW ---
st.header("Retention Candidates Overview")
st.markdown("These are the customers identified as high-risk for churning. The table shows their churn probability, Customer Lifetime Value (CLV), and the recommended retention strategy.")

try:
    paths = ['../Dataset/retention_candidates.csv', 'Dataset/retention_candidates.csv']

    for path in paths:
        if os.path.exists(path):
         retention_df = pd.read_csv(path)
         break

    churning_customers = retention_df[retention_df['churn_prediction'] == 1].copy()

    if churning_customers.empty:
        st.warning("No customers are currently predicted to churn. Great news!")
    else:
        # Load retention strategies (assuming the logic from retention_strategy.py)
        def get_retention_strategy(row):
            churn_drivers = row['top_churn_drivers']
            clv_tier = row.get('clv_tier', 'Medium') # Default to medium if not present
            
            if clv_tier == 'High':
                base_strategy = "Offer a 15% discount and a free premium service for 3 months."
            elif clv_tier == 'Medium':
                base_strategy = "Offer a 10% discount on the next bill."
            else:
                base_strategy = "Send a survey to understand their dissatisfaction."

            if 'tenure_monthly_ratio' in churn_drivers:
                return base_strategy + " Also, offer a longer-term contract with a lower monthly rate."
            elif 'InternetService_Fiber optic' in churn_drivers:
                return base_strategy + " Also, schedule a free technical check-up."
            else:
                return base_strategy

        churning_customers['retention_strategy'] = churning_customers.apply(get_retention_strategy, axis=1)
        
        st.dataframe(churning_customers[['customerID', 'clv', 'clv_tier', 'churn_probability', 'retention_strategy']])

except FileNotFoundError:
    st.error("The `retention_candidates.csv` file was not found. Please run the prediction pipeline first.")
except Exception as e:
    st.error(f"An error occurred while loading the retention data: {e}")


# --- 3. REAL-TIME CHURN PREDICTION ---
st.header("Real-time Churn Prediction")
st.markdown("Enter a new customer's details below to get an instant churn prediction from the API.")

# Create a form for user input
with st.form(key='prediction_form'):
    st.subheader("Customer Details")
    
    # Split layout into columns for better organization
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=1.0)

    with col2:
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        TotalCharges = st.number_input("Total Charges", min_value=0.0, value=1400.0, step=10.0)

    with col3:
        tenure = st.slider("Tenure (months)", 1, 72, 24)
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    submit_button = st.form_submit_button(label='Predict Churn')

# --- 4. DISPLAY PREDICTION RESULTS ---
if submit_button:
    # Prepare the data in the format the API expects
    customer_data = {
        "gender": gender, "SeniorCitizen": SeniorCitizen, "Partner": Partner,
        "Dependents": Dependents, "tenure": tenure, "PhoneService": PhoneService,
        "MultipleLines": MultipleLines, "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection, "TechSupport": TechSupport,
        "StreamingTV": StreamingTV, "StreamingMovies": StreamingMovies,
        "Contract": Contract, "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod, "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    # Send the request to the FastAPI endpoint
    api_url = "http://127.0.0.1:8000/predict"
    try:
        response = requests.post(api_url, data=json.dumps(customer_data))
        response.raise_for_status()  # Raise an exception for bad status codes
        
        result = response.json()
        
        st.subheader("Prediction Result")
        
        churn_prob = result['churn_probability']
        
        if result['churn_prediction'] == 1:
            st.error(f"This customer is likely to CHURN with a probability of {churn_prob:.2%}.")
        else:
            st.success(f"This customer is likely to STAY with a probability of {1-churn_prob:.2%}.")
            
        st.subheader("Top Churn Drivers")
        st.info("These are the top 3 factors influencing the prediction:")
        for driver in result['top_churn_drivers']:
            st.write(f"- {driver.replace('_', ' ').title()}")

    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the API. Please ensure the FastAPI server is running. Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
