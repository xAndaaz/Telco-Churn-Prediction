import os
import streamlit as st
import pandas as pd
import requests
import json
import sys

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- 1. HEADER ---
st.title("Churn Prediction and Retention Dashboard")
st.markdown("This dashboard provides insights into customer churn predictions and helps generate retention strategies.")

# --- 3. BATCH PREDICTION FROM CSV ---
st.header("Batch Prediction from CSV")
st.markdown("Upload a CSV file with customer data to get predictions and retention strategies for the entire list.")

# Import pipeline functions here to avoid running them on every page load
# Ensure the parent directory is in sys.path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prediction_pipeline import run_prediction_pipeline
from survival_prediction_pipeline import run_survival_prediction_pipeline
from survival_retention_strategy import generate_survival_retention_strategies


uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    with st.spinner('Processing customer list... This may take a moment.'):
        try:
            input_df = pd.read_csv(uploaded_file)
            
            # Run the full pipeline
            results_df = run_prediction_pipeline(input_df)
            
            st.success("Processing complete!")
            
            # --- Display Summary Metrics ---
            total_customers = len(results_df)
            churner_count = results_df['churn_prediction'].sum()
            st.metric(label="Total Customers Processed", value=total_customers)
            st.metric(label="At-Risk Customers Identified", value=churner_count)
            
            st.subheader("Prediction Results")
            st.markdown("The table below shows the key results. Use the download button for the full dataset.")

            # --- Create a focused view for the dashboard ---
            # Check for customerID, as it might not be in every uploaded file
            display_columns = ['churn_prediction', 'churn_probability', 'retention_strategy', 'top_churn_drivers']
            if 'customerID' in results_df.columns:
                display_columns.insert(0, 'customerID')

            st.dataframe(results_df[display_columns])
            
            # --- Provide a download button for the FULL results ---
            
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv_output = convert_df_to_csv(results_df)
            
            st.download_button(
                label="Download Full Results as CSV",
                data=csv_output,
                file_name='retention_candidates_output.csv',
                mime='text/csv',
            )
            
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

# --- NEW: SURVIVAL ANALYSIS SECTION ---
st.header("Survival Analysis & Time-Based Retention")
st.markdown("Run the survival analysis pipeline on the sample data to generate time-sensitive retention strategies.")

if st.button("Run Survival Analysis on Sample Data"):
    with st.spinner("Running survival analysis... This may take a few moments."):
        try:
            # Run the prediction pipeline
            survival_predictions_path = run_survival_prediction_pipeline()
            
            if survival_predictions_path:
                # Load the predictions and generate strategies
                predictions_df = pd.read_csv(survival_predictions_path)
                retention_plan_df = generate_survival_retention_strategies(predictions_df)
                
                st.success("Survival analysis complete!")
                
                st.subheader("Time-Based Retention Plan")
                st.markdown("The table below shows the recommended retention strategy based on churn risk over time.")
                
                # Display the results
                st.dataframe(retention_plan_df[['customerID', 'retention_strategy']])
                
                # Provide a download button for the survival retention plan
                csv_output_survival = convert_df_to_csv(retention_plan_df)
                st.download_button(
                    label="Download Survival Retention Plan as CSV",
                    data=csv_output_survival,
                    file_name='survival_retention_plan.csv',
                    mime='text/csv',
                )
        except Exception as e:
            st.error(f"An error occurred during survival analysis: {e}")


# --- 4. REAL-TIME CHURN PREDICTION ---
st.header("Real-time Single Prediction")
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
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=120.0, step=1.0)

    with col2:
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        TotalCharges = st.number_input("Total Charges", min_value=0.0, value=240.0, step=10.0)

    with col3:
        tenure = st.slider("Tenure (months)", 1, 72, 2)
        InternetService = st.selectbox("Internet Service", ["Fiber optic", "DSL",  "No"])
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    submit_button = st.form_submit_button(label='Predict Churn')

# --- 4. DISPLAY PREDICTION RESULTS ---
if submit_button:
    # Add the parent directory to the Python path to allow imports
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from retention_strategy import get_retention_strategies

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
            
        st.subheader("Recommended Retention Strategies")
        st.info("Based on the key drivers, here are some recommended actions:")

        # Add premium services count for more detailed strategies
        premium_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
        customer_data['premium_services_count'] = sum(1 for service in premium_services if customer_data[service] == 'Yes')
        
        strategies = get_retention_strategies(customer_data, result['top_churn_drivers'])
        for i, strategy in enumerate(strategies, 1):
            st.write(f"**{i}.** {strategy}")

    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the API. Please ensure the FastAPI server is running. Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
