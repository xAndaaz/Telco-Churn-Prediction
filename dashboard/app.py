import os
import streamlit as st
import pandas as pd
import requests
import json
import sys
import subprocess
import pickle
import matplotlib.pyplot as plt
import shap

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- INITIALIZE SESSION STATE ---
# This ensures that our variables persist across reruns
if 'batch_results_df' not in st.session_state:
    st.session_state['batch_results_df'] = None
if 'single_prediction_result' not in st.session_state:
    st.session_state['single_prediction_result'] = None

# --- HELPER FUNCTIONS ---
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string for downloading."""
    return df.to_csv(index=False).encode('utf-8')

# --- UI: HEADER ---
st.title("Churn Prediction and Risk Profile Dashboard")
st.markdown("An end-to-end tool to identify at-risk customers and understand the 'why' and 'when' of their churn risk.")

# --- UI: BATCH ANALYSIS ---
st.header("Generate Churn Risk Profiles from a CSV")
st.markdown("Upload a CSV file with customer data to run the full analysis pipeline.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    if st.button("Generate Churn Risk Profiles"):
        # Save the uploaded file to a temporary location
        temp_input_path = os.path.join('Dataset', 'temp_uploaded_data.csv')
        input_df = pd.read_csv(uploaded_file)
        input_df.to_csv(temp_input_path, index=False)
        
        with st.spinner('Running end-to-end analysis... This may take a moment.'):
            try:
                python_executable = os.path.join('.venv', 'Scripts', 'python.exe')
                
                # Run all pipelines
                st.info("Step 1/3: Running classification pipeline...")
                subprocess.run([python_executable, 'prediction_pipeline.py', '--input', temp_input_path], check=True, capture_output=True, text=True)
                st.info("Step 2/3: Running survival analysis pipeline...")
                subprocess.run([python_executable, 'survival_prediction_pipeline.py', '--input', temp_input_path], check=True, capture_output=True, text=True)
                subprocess.run([python_executable, 'survival_risk_analyzer.py'], check=True, capture_output=True, text=True)
                st.info("Step 3/3: Generating unified churn risk profiles...")
                subprocess.run([python_executable, 'master_retention_pipeline.py'], check=True, capture_output=True, text=True)
                
                # Load and store results in session state
                st.session_state['batch_results_df'] = pd.read_csv('Dataset/master_retention_plan.csv')
                st.success("Analysis complete!")

            except subprocess.CalledProcessError as e:
                st.error(f"An error occurred while running the pipeline: {e.stderr}")
            except FileNotFoundError:
                st.error("Could not find one of the required pipeline scripts.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# --- DISPLAY BATCH RESULTS (if they exist in session state) ---
if st.session_state['batch_results_df'] is not None:
    st.header("Batch Analysis Results")
    results_df = st.session_state['batch_results_df']
    
    # Display metrics
    total_customers = len(results_df)
    churner_count = results_df['churn_prediction'].sum()
    st.metric(label="Total Customers Processed", value=total_customers)
    st.metric(label="At-Risk Customers Identified", value=churner_count)
    
    # Display results table
    st.subheader("Churn Risk Profiles")
    display_columns = [
        'customerID', 'churn_prediction', 'ProbabilityRiskTier', 
        'TimeBasedRisk', 'ActionableInsight', 'clv_tier', 'churn_probability'
    ]
    st.dataframe(results_df[display_columns])
    
    # Display download button
    csv_output = convert_df_to_csv(results_df)
    st.download_button(
        label="Download Full Profiles as CSV",
        data=csv_output,
        file_name='master_retention_plan.csv',
        mime='text/csv',
    )

    # Display SHAP plot
    st.subheader("Feature Impact on Churn Prediction (Overall)")
    st.markdown("""
    This beeswarm plot shows the impact of the top features on the model's output for the entire dataset.
    - **Each dot** is a single customer from your uploaded file.
    - **The x-axis** is the SHAP value: a positive value means the feature pushes the prediction towards churn (1), while a negative value pushes it away from churn (0).
    - **The color** shows the feature's value for that customer: red for high values, blue for low values.
    """)
    try:
        with open('Models/shap_values.pkl', 'rb') as f:
            shap_values = pickle.load(f)
        with open('Models/prepared_data.pkl', 'rb') as f:
            prepared_data = pickle.load(f)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, prepared_data, plot_type="dot", show=False)
        plt.tight_layout()
        st.pyplot(fig)
    except FileNotFoundError:
        st.warning("Could not generate SHAP summary plot. Required files not found.")
    except Exception as e:
        st.error(f"An error occurred while generating the SHAP plot: {e}")

# --- UI: REAL-TIME ANALYSIS ---
st.header("Real-time Single Customer Analysis")
st.markdown("Enter a customer's details below to get an instant risk analysis.")

with st.form(key='prediction_form'):
    st.subheader("Customer Details")
    col1, col2, col3 = st.columns(3)
    # Form inputs... (same as before)
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
        TotalCharges = st.number_input("Total Charges", min_value=0.0, value=350.0, step=10.0)
    with col3:
        tenure = st.slider("Tenure (months)", 1, 72, 5)
        InternetService = st.selectbox("Internet Service", ["Fiber optic", "DSL",  "No"])
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    
    submit_button = st.form_submit_button(label='Analyze Churn Risk')

    if submit_button:
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
        
        api_url = "http://127.0.0.1:8000/predict"
        try:
            response = requests.post(api_url, data=json.dumps(customer_data))
            response.raise_for_status()
            # Store result in session state
            st.session_state['single_prediction_result'] = response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the API. Please ensure the FastAPI server is running. Error: {e}")
            st.session_state['single_prediction_result'] = None
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.session_state['single_prediction_result'] = None

# --- DISPLAY SINGLE PREDICTION RESULT (if it exists in session state) ---
if st.session_state['single_prediction_result'] is not None:
    st.header("Single Customer Analysis Result")
    result = st.session_state['single_prediction_result']
    
    churn_prob = result['churn_probability']
    if result['churn_prediction'] == 1:
        st.error(f"This customer is **AT RISK** of churning with a probability score of {churn_prob:.2f}.")
    else:
        st.success(f"This customer is **NOT** at risk of churning (Probability Score: {churn_prob:.2f}).")
        
    st.subheader("Key Churn Drivers")
    st.info("The following factors were the most influential in this prediction:")
    for i, driver in enumerate(result['top_churn_drivers'], 1):
        st.write(f"**{i}.** {driver.replace('_', ' ').title()}")
