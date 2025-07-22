import os
import streamlit as st
import pandas as pd
import requests
import json
import sys
import threading
import uvicorn
import pickle
import matplotlib.pyplot as plt
import shap

# Add project root to path to allow module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import necessary components from the project
from api.main import app as fastapi_app
from churnadvisor.pipelines.prediction_pipeline import run_prediction_pipeline
from churnadvisor.pipelines.survival_prediction_pipeline import run_survival_prediction_pipeline
from churnadvisor.pipelines.survival_risk_analyzer import generate_time_based_risk
from churnadvisor.pipelines.master_retention_pipeline import generate_churn_risk_profiles

# --- FastAPI Server in Background Thread ---
def run_api():
    """Runs the FastAPI server in a separate thread."""
    uvicorn.run(fastapi_app, host="127.0.0.1", port=8000)

# Use session state to ensure the server thread is only started once
if "server_thread" not in st.session_state:
    st.session_state.server_thread = threading.Thread(target=run_api, daemon=True)
    st.session_state.server_thread.start()
# -----------------------------------------

st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- SESSION STATE INITIALIZATION ---
if 'page' not in st.session_state:
    st.session_state['page'] = 'Batch Analysis'
if 'batch_results_df' not in st.session_state:
    st.session_state['batch_results_df'] = None
if 'single_prediction_result' not in st.session_state:
    st.session_state['single_prediction_result'] = None

def convert_df_to_csv(df):
    """Utility to convert DataFrame to CSV for download."""
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

st.title(f"🔍 {st.session_state['page']}")

# --- BATCH ANALYSIS PAGE ---
if st.session_state['page'] == 'Batch Analysis':
    st.markdown("Upload a CSV file with customer data to run the full analysis pipeline and generate churn risk profiles for all customers.")
    
    with st.container(border=True):
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            if st.button("Generate Churn Risk Profiles", use_container_width=True):
                
                with st.spinner('Running end-to-end analysis... This may take a moment.'):
                    try:
                        # Load the uploaded data into a DataFrame
                        source_df = pd.read_csv(uploaded_file)

                        # --- Pipeline Orchestration (In-Memory) ---
                        st.info("Step 1/3: Running classification pipeline...")
                        prediction_results_df = run_prediction_pipeline(source_df)

                        st.info("Step 2/3: Running survival analysis pipeline...")
                        survival_predictions_df = run_survival_prediction_pipeline(source_df)
                        survival_risk_df = generate_time_based_risk(survival_predictions_df)

                        st.info("Step 3/3: Generating unified churn risk profiles...")
                        # Call the refactored function with the dataframes in memory
                        final_results_df = generate_churn_risk_profiles(prediction_results_df, survival_risk_df)
                        
                        # Store the final results in the session state
                        st.session_state['batch_results_df'] = final_results_df
                        st.success("Analysis complete!")

                    except Exception as e:
                        st.error(f"An error occurred during the pipeline execution.")
                        st.exception(e)

    # --- Display Batch Results ---
    if st.session_state['batch_results_df'] is not None:
        st.markdown("---")
        st.subheader("📈 Batch Analysis Results")
        results_df = st.session_state['batch_results_df']
        
        col1, col2 = st.columns(2);
        with col1:
            st.metric(label="Total Customers Processed", value=len(results_df))
        with col2:
            st.metric(label="At-Risk Customers Identified", value=results_df['churn_prediction'].sum())

        with st.container(border=True):
            st.subheader("📄 At-Risk Customer Overview")
            st.markdown("This table provides a high-level, scrollable overview of all at-risk customers. Select a customer from the dropdown menu below to see their detailed, unabridged **Actionable Insight**.")
            
            at_risk_df = results_df[results_df['churn_prediction'] == 1].copy()
            
            if not at_risk_df.empty:
                summary_columns = ['customerID', 'ProbabilityRiskTier', 'TimeBasedRisk', 'clv_tier', 'ActionableInsight']
                st.dataframe(at_risk_df[summary_columns])

                st.markdown("---")
                selected_customer = st.selectbox(
                    '**Select a Customer ID to View Detailed Insight:**',
                    options=at_risk_df['customerID'].tolist()
                )
                
                if selected_customer:
                    customer_insight = at_risk_df[at_risk_df['customerID'] == selected_customer]['ActionableInsight'].iloc[0]
                    st.info(f"**Actionable Insight for {selected_customer}:**\n\n{customer_insight}")
            else:
                st.success("✅ No customers were identified as being at risk of churn in the uploaded file.")

            st.download_button(
                label="Download Full Profiles for All Customers as CSV",
                data=convert_df_to_csv(results_df),
                file_name='master_retention_plan.csv',
                mime='text/csv',
                use_container_width=True
            )

        with st.container(border=True):
            st.subheader("📊 Feature Impact on Churn Prediction (Overall)")
            st.markdown("This beeswarm plot shows the impact of the top features on the model's output for the entire dataset.")
            try:
                with open('Models/shap_values.pkl', 'rb') as f: shap_values_data = pickle.load(f)
                with open('Models/prepared_data.pkl', 'rb') as f: prepared_data = pickle.load(f)
                
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values_data, prepared_data, plot_type="dot", show=False)
                plot_path = 'shap_summary.png'
                fig.savefig(plot_path, bbox_inches='tight', dpi=150)
                plt.close(fig)
                st.image(plot_path, width=800)
            except FileNotFoundError:
                st.warning("SHAP plot data not found. Please run a batch analysis first.")
            except Exception as e:
                st.error(f"An error occurred while generating the SHAP plot: {e}")

# --- INSTANT PREDICTION PAGE ---
elif st.session_state['page'] == 'Instant Prediction':
    st.markdown("Enter a customer's details below to get an instant risk analysis.")
    
    with st.form(key='prediction_form'):
        st.subheader("📝 Customer Details")
        col1, col2, col3 = st.columns(3);
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
        b_col1, b_col2, b_col3 = st.columns(3);
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
            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the prediction API. Please ensure it's running. Error: {e}")
                st.session_state['single_prediction_result'] = None
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.session_state['single_prediction_result'] = None

    # --- Display Single Prediction Result ---
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
            drivers = result.get('top_churn_drivers', [])
            if isinstance(drivers, list):
                for i, driver in enumerate(drivers, 1):
                    st.write(f"**{i}.** {driver.replace('_', ' ').title()}")
            else:
                st.warning("Could not determine key churn drivers.")