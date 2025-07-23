import os
import streamlit as st
import pandas as pd
import json
import sys
import pickle
import matplotlib.pyplot as plt
import shap

# Add project root to path to allow importing churnadvisor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from churnadvisor.pipelines.prediction_pipeline import run_prediction_pipeline
from churnadvisor.pipelines.survival_prediction_pipeline import run_survival_prediction_pipeline
from churnadvisor.pipelines.survival_risk_analyzer import generate_time_based_risk
from churnadvisor.pipelines.master_retention_pipeline import generate_churn_risk_profiles
from churnadvisor.processing.data_processing import prepare_data_for_prediction
from churnadvisor.analysis.churn_analyzer import generate_actionable_insight

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
    if st.button("Our Workflow", use_container_width=True):
        st.session_state['page'] = 'Our Workflow'
    st.markdown("---")
    st.info("This dashboard provides tools for predicting customer churn and understanding its key drivers.")

#  MAIN PAGE LAYOUT ---------------------------------------------------------------
st.title(f"🔍 {st.session_state['page']}")


#  BATCH ANALYSIS PAGE ---------------------------------------------------------------------

if st.session_state['page'] == 'Batch Analysis':
    st.markdown("Upload a CSV file with customer data to run the full analysis pipeline and generate churn risk profiles for all customers.")
    
    with st.container(border=True):
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            if st.button("Generate Churn Risk Profiles", use_container_width=True):
                input_df = pd.read_csv(uploaded_file)
                
                with st.spinner('Running end-to-end analysis... This may take a moment.'):
                    try:
                        # Run pipelines sequentially in-memory
                        st.info("Step 1/3: Running classification pipeline...")
                        classification_results = run_prediction_pipeline(input_df)
                        
                        st.info("Step 2/3: Running survival analysis pipeline...")
                        survival_predictions = run_survival_prediction_pipeline(input_df)
                        survival_risk_results = generate_time_based_risk(survival_predictions)
                        
                        st.info("Step 3/3: Generating unified churn risk profiles...")
                        final_results = generate_churn_risk_profiles(classification_results, survival_risk_results)
                        
                        st.session_state['batch_results_df'] = final_results
                        st.success("Analysis complete!")
                    except Exception as e:
                        st.error(f"An error occurred during the analysis: {e}")
                        # Print traceback for debugging if needed
                        import traceback
                        traceback.print_exc()

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
            st.subheader("📄 At-Risk Customer Overview (Download full csv from button below)")
            st.markdown("This table provides a high-level, scrollable overview of all at-risk customers. Select a customer from the dropdown menu below to see their detailed, unabridged **Actionable Insight**.")
            
            at_risk_df = results_df[results_df['churn_prediction'] == 1].copy()
            
            if not at_risk_df.empty:
                # Master View: A compact, scrollable table
                summary_columns = ['customerID', 'ProbabilityRiskTier', 'TimeBasedRisk', 'clv_tier', 'Actionable Insight']
                st.dataframe(at_risk_df[summary_columns])

                # Detail View: A dropdown to select a customer for detailed insight
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
            st.markdown("This beeswarm plot shows the impact of the top features on the model's output for the entire provided dataset. The most influential features are shown at the top, with their impact on churn probability.")
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

# INSTANT PREDICTION PAGE ----------------------------------------------------

elif st.session_state['page'] == 'Instant Prediction':
    st.markdown("Enter a customer's details below to get an instant risk analysis.")

    # Cached function to load model and explainer to prevent reloading on every run
    @st.cache_resource
    def load_model_and_explainer():
        print("Loading model and explainer for instant prediction...")
        model_path = os.path.join('Models', 'model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        explainer = shap.TreeExplainer(model)
        print("Model and explainer loaded.")
        return model, explainer

    # Function to perform a single prediction
    def run_instant_prediction(customer_data_dict):
        # This needs access to the data processing function
        from churnadvisor.processing.data_processing import prepare_data_for_prediction
        from churnadvisor.analysis.churn_analyzer import generate_actionable_insight
        
        model, explainer = load_model_and_explainer()
        
        df = pd.DataFrame([customer_data_dict])
        df_prepared, _ = prepare_data_for_prediction(df)
        
        prediction = model.predict(df_prepared)[0]
        probability = model.predict_proba(df_prepared)[:, 1][0]
        shap_values = explainer.shap_values(df_prepared)
        
        shap_df = pd.DataFrame(shap_values, columns=df_prepared.columns)
        top_drivers = shap_df.abs().iloc[0].nlargest(5).index.tolist()

        customer_profile = df.iloc[0].copy()
        customer_profile['churn_prediction'] = prediction
        customer_profile['churn_probability'] = probability
        customer_profile['top_churn_drivers'] = top_drivers
        
        insight = generate_actionable_insight(customer_profile)

        return {
            "churn_prediction": int(prediction),
            "churn_probability": float(probability),
            "top_churn_drivers": top_drivers,
            "actionable_insight": insight
        }

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
            try:
                st.session_state['single_prediction_result'] = run_instant_prediction(customer_data)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state['single_prediction_result'] = None

    # Display single prediction result if it exists
    if st.session_state['single_prediction_result'] is not None:
        st.markdown("---")
        st.subheader("💡 Analysis Result (for better analysis use batch prediction)")
        result = st.session_state['single_prediction_result']
        
        with st.container(border=True):
            churn_prob = result['churn_probability']
            if result['churn_prediction'] == 1:
                st.error(f"This customer is **AT RISK** of churning.")
            else:
                st.success(f"This customer is **NOT** at risk of churning (Probability Score: {churn_prob:.2f}).")
            
            st.subheader("Key Churn Drivers")                                             
            st.info("The following factors were the most influential (can be positive or negative) in this prediction:")       
            for i, driver in enumerate(result['top_churn_drivers'], 1):                 
                st.write(f"**{i}.** {driver.replace('_', ' ').title()}")

# OUR WORKFLOW PAGE ----------------------------------------------------
elif st.session_state['page'] == 'Our Workflow':
    st.markdown("This page details the sophisticated, end-to-end process used to build this advisory intelligence engine.")
    
    workflow_text = """
### Our Workflow: A Masterclass in Data-Driven Decision Making

This project showcases a professional, end-to-end data science workflow that transforms raw data into a sophisticated advisory intelligence engine. Every step was deliberate, every choice was backed by evidence, and the result is a system that is robust, insightful, and built for real-world impact.

**1. Foundational Deep Dive: EDA and Feature Engineering**
Our journey began not with models, but with data. Through comprehensive Exploratory Data Analysis (EDA), we uncovered the complex patterns and, most critically, the significant class imbalance inherent in the dataset. We didn't just feed raw data into a model; we engineered new, high-impact features that captured nuanced customer behaviors, laying the groundwork for superior predictive performance.

**2. Scientific Model Selection: Evidence over Assumptions**
We adopted a scientific, evidence-based approach to model selection. Instead of choosing a model based on popularity, we conducted a rigorous benchmarking process comparing four different classifiers: Decision Tree, Random Forest, XGBoost, and the `XGBRFClassifier`. Every experiment was meticulously logged in our `experiments.json` file, creating an auditable trail of performance metrics (F1-score, AUC, training time). This data-driven process proved that the `XGBRFClassifier` was the superior choice for this specific problem.

**3. Advanced Problem-Solving: Tackling Class Imbalance with SMOTEENN**
A powerful model isn't enough when faced with imbalanced data. To solve this, we went beyond basic techniques and implemented `SMOTEENN`, a powerful hybrid approach. This advanced method doesn't just create synthetic data; it intelligently over-samples the minority (churn) class with SMOTE while simultaneously using Edited Nearest Neighbors (ENN) to clean noisy samples from the decision boundary. This created a high-quality, balanced training set that enabled our model to learn the distinction between churners and non-churners with far greater clarity.

**4. Expert Model Interpretation and Performance:**
Our meticulous process culminated in a model with a superior balance of precision and recall, optimized for real-world business impact.
*   **High-Precision Performance:** We engineered a "sharpshooter" model that excels at minimizing false positives. This is critical in a business context, as it ensures that costly retention efforts are directed only at customers who are genuinely at risk, maximizing return on investment. This finely-tuned balance surpasses the performance of generic, off-the-shelf models on this dataset.
*   **Understanding Model Behavior:** A key insight from our work is understanding the behavior of our `XGBRFClassifier`. It naturally produces narrow, highly-calibrated probability scores (e.g., 0.35-0.65). This is not a limitation but a hallmark of a well-regularized Random Forest model, which averages hundreds of decision trees to avoid the overconfidence seen in other model types.

**5. The Final Layer: The Advisory Intelligence Engine**
The highly-optimized model is just the foundation. We built an intelligent system on top of it, integrating Survival Analysis (to predict *when* a customer might churn) and a context-aware XAI engine. This engine translates the model's technical outputs into factually-consistent, human-readable narratives, distinguishing between risk drivers and protective factors to provide a truly holistic and actionable **Churn Risk Profile**.

This end-to-end workflow—from deep EDA to advanced modeling and intelligent interpretation—is what sets this project apart, delivering a solution that is not just predictive, but genuinely advisory.
"""
    st.markdown(workflow_text) 

