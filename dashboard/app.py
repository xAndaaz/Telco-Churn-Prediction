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
if 'page' not in st.session_state:
    st.session_state['page'] = 'Batch Analysis'
if 'batch_results_df' not in st.session_state:
    st.session_state['batch_results_df'] = None
if 'single_prediction_result' not in st.session_state:
    st.session_state['single_prediction_result'] = None

# --- HELPER FUNCTIONS ---
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("📊 ChurnAdvisor")
    st.markdown("---")
    if st.button("Batch Analysis", use_container_width=True):
        st.session_state['page'] = 'Batch Analysis'
    if st.button("Instant Prediction", use_container_width=True):
        st.session_state['page'] = 'Instant Prediction'
    st.markdown("---")
    st.info("This dashboard provides tools for predicting customer churn and understanding its key drivers.")

# --- MAIN PAGE LAYOUT ---
st.title(f"🔍 {st.session_state['page']}")

# =================================================================================================
# --- BATCH ANALYSIS PAGE ---
# =================================================================================================
if st.session_state['page'] == 'Batch Analysis':
    st.markdown("Upload a CSV file with customer data to run the full analysis pipeline and generate churn risk profiles for all customers.")
    
    with st.container(border=True):
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            if st.button("Generate Churn Risk Profiles", use_container_width=True):
                temp_input_path = os.path.join('Dataset', 'temp_uploaded_data.csv')
                pd.read_csv(uploaded_file).to_csv(temp_input_path, index=False)
                
                with st.spinner('Running end-to-end analysis... This may take a moment.'):
                    try:
                        python_executable = os.path.join('.venv', 'Scripts', 'python.exe')
                        # Run pipelines
                        st.info("Step 1/3: Running classification pipeline...")
                        subprocess.run([python_executable, 'prediction_pipeline.py', '--input', temp_input_path], check=True, capture_output=True, text=True)
                        st.info("Step 2/3: Running survival analysis pipeline...")
                        subprocess.run([python_executable, 'survival_prediction_pipeline.py', '--input', temp_input_path], check=True, capture_output=True, text=True)
                        subprocess.run([python_executable, 'survival_risk_analyzer.py'], check=True, capture_output=True, text=True)
                        st.info("Step 3/3: Generating unified churn risk profiles...")
                        subprocess.run([python_executable, 'master_retention_pipeline.py'], check=True, capture_output=True, text=True)
                        
                        st.session_state['batch_results_df'] = pd.read_csv('Dataset/master_retention_plan.csv')
                        st.success("Analysis complete!")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

    # Display batch results if they exist
    if st.session_state['batch_results_df'] is not None:
        st.markdown("---")
        st.subheader("📈 Batch Analysis Results")
        results_df = st.session_state['batch_results_df']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Total Customers Processed", value=len(results_df))
        with col2:
            st.metric(label="At-Risk Customers Identified", value=results_df['churn_prediction'].sum())

        with st.container(border=True):
            st.subheader("📄 Churn Risk Profiles")
            display_columns = ['customerID', 'churn_prediction', 'ProbabilityRiskTier', 'TimeBasedRisk', 'ActionableInsight', 'clv_tier', 'churn_probability']
            st.dataframe(results_df[display_columns])
            st.download_button(
                label="Download Full Profiles as CSV",
                data=convert_df_to_csv(results_df),
                file_name='master_retention_plan.csv',
                mime='text/csv',
                use_container_width=True
            )

        with st.container(border=True):
            st.subheader("📊 Feature Impact on Churn Prediction (Overall)")
            st.markdown("This beeswarm plot shows the impact of the top features on the model's output for the entire dataset.")
            try:
                with open('Models/shap_values.pkl', 'rb') as f: shap_values = pickle.load(f)
                with open('Models/prepared_data.pkl', 'rb') as f: prepared_data = pickle.load(f)
                
                plt.figure()
                shap.summary_plot(shap_values, prepared_data, plot_type="dot", show=False)
                plot_path = 'shap_summary.png'
                plt.savefig(plot_path, bbox_inches='tight', dpi=150)
                plt.close()
                st.image(plot_path, width=800)
            except Exception as e:
                st.error(f"An error occurred while generating the SHAP plot: {e}")

# =================================================================================================
# --- INSTANT PREDICTION PAGE ---
# =================================================================================================
elif st.session_state['page'] == 'Instant Prediction':
    st.markdown("Enter a customer's details below to get an instant risk analysis.")
    
    with st.form(key='prediction_form'):
        st.subheader("📝 Customer Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            Partner = st.selectbox("Partner", ["Yes", "No"])
            PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
            StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with col2:
            SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            Dependents = st.selectbox("Dependents", ["Yes", "No"])
            MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        with col3:
            tenure = st.slider("Tenure (months)", 1, 72, 5)
            InternetService = st.selectbox("Internet Service", ["Fiber optic", "DSL",  "No"])
            OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        
        st.subheader("Billing Information")
        b_col1, b_col2, b_col3 = st.columns(3)
        with b_col1:
            MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=1.0)
        with b_col2:
            TotalCharges = st.number_input("Total Charges", min_value=0.0, value=350.0, step=10.0)
        with b_col3:
            PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

        submit_button = st.form_submit_button(label='Analyze Churn Risk', use_container_width=True)

        if submit_button:
            customer_data = {
                "gender": gender, "SeniorCitizen": SeniorCitizen, "Partner": Partner, "Dependents": Dependents, "tenure": tenure, 
                "PhoneService": PhoneService, "MultipleLines": MultipleLines, "InternetService": InternetService, "OnlineSecurity": OnlineSecurity,
                "OnlineBackup": OnlineBackup, "DeviceProtection": DeviceProtection, "TechSupport": TechSupport, "StreamingTV": StreamingTV,
                "StreamingMovies": StreamingMovies, "Contract": Contract, "PaperlessBilling": PaperlessBilling, "PaymentMethod": PaymentMethod,
                "MonthlyCharges": MonthlyCharges, "TotalCharges": TotalCharges
            }
            api_url = "http://127.0.0.1:8000/predict"
            try:
                response = requests.post(api_url, data=json.dumps(customer_data))
                response.raise_for_status()
                st.session_state['single_prediction_result'] = response.json()
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state['single_prediction_result'] = None

    # Display single prediction result if it exists
    if st.session_state['single_prediction_result'] is not None:
        st.markdown("---")
        st.subheader("💡 Analysis Result")
        result = st.session_state['single_prediction_result']
        
        with st.container(border=True):
            churn_prob = result['churn_probability']
            if result['churn_prediction'] == 1:
                st.error(f"This customer is **AT RISK** of churning with a probability score of **{churn_prob:.2f}**.")
            else:
                st.success(f"This customer is **NOT** at risk of churning (Probability Score: {churn_prob:.2f}).")
            
            st.subheader("Key Churn Drivers")
            st.info("The following factors were the most influential in this prediction:")
            for i, driver in enumerate(result['top_churn_drivers'], 1):
                st.write(f"**{i}.** {driver.replace('_', ' ').title()}")

