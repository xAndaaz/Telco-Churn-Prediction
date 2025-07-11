import pandas as pd
import pickle
import shap
from retention_strategy import get_retention_strategies

# --- 1. LOAD MODELS AND COLUMNS (Load once to be used by functions) ---
try:
    with open('Models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('Models/training_columns.pkl', 'rb') as f:
        training_columns = pickle.load(f)
    with open('Models/clv_bins.pkl', 'rb') as f:
        clv_bins = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print("Please ensure you have run the `train_model.py` script to generate the necessary model files.")
    # Exit if models can't be loaded, as the script is unusable.
    exit()

# --- 2. CORE LOGIC FUNCTIONS ---

def prepare_data_for_prediction(df):
    """Prepares raw data for prediction by applying transformations from training."""
    
    # Ensure required columns exist before calculations
    required_cols = ['MonthlyCharges', 'tenure', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in uploaded CSV: {col}")

    # Use the same CLV calculation as in training
    assumed_acquisition_cost = 100
    df['clv'] = (df['MonthlyCharges'] * df['tenure']) - assumed_acquisition_cost
    
    # Use the pre-calculated bins from training to categorize CLV
    df['clv_tier'] = pd.cut(df['clv'], bins=clv_bins, labels=['Low', 'Medium', 'High'], include_lowest=True)
    
    # Recreate the same engineered features
    df['tenure_monthly_interaction'] = df['tenure'] * df['MonthlyCharges']
    premium_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    df['premium_services_count'] = df[premium_services].apply(lambda x: (x == 'Yes').sum(), axis=1)
    df['tenure_monthly_ratio'] = df['tenure'] / (df['MonthlyCharges'] + 1e-6)
    
    # One-hot encode categorical features
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Align columns with the training data
    df_aligned = df_encoded.reindex(columns=training_columns, fill_value=0)
    
    return df_aligned

def run_prediction_pipeline(df):
    """
    Runs the full prediction and explanation pipeline on a dataframe.
    
    Args:
        df (pd.DataFrame): A dataframe with customer data.
        
    Returns:
        pd.DataFrame: The original dataframe with added columns for predictions and strategies.
    """
    
    # Prepare the data
    df_prepared = prepare_data_for_prediction(df.copy())
    
    # Get predictions
    predictions = model.predict(df_prepared)
    probabilities = model.predict_proba(df_prepared)[:, 1]
    
    # Get SHAP explanations for top drivers
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_prepared)
    shap_df = pd.DataFrame(shap_values, columns=df_prepared.columns)
    top_features = shap_df.abs().apply(lambda x: x.nlargest(3).index.tolist(), axis=1)
    
    # Combine results into a single DataFrame
    results = df.copy()
    results['churn_prediction'] = predictions
    results['churn_probability'] = probabilities
    results['top_churn_drivers'] = top_features
    
    # Generate retention strategies for customers predicted to churn
    churning_customers_mask = results['churn_prediction'] == 1
    if churning_customers_mask.any():
        strategies = []
        for _, row in results[churning_customers_mask].iterrows():
            customer_data = row.to_dict()
            drivers = row['top_churn_drivers']
            strategy = get_retention_strategies(customer_data, drivers)
            strategies.append(" | ".join(strategy))
        
        results.loc[churning_customers_mask, 'retention_strategy'] = strategies
    
    # Fill strategy for non-churners
    results['retention_strategy'].fillna("No action needed", inplace=True)
    
    return results

# --- 3. SCRIPT EXECUTION (for command-line use) ---

if __name__ == '__main__':
    print("Running prediction pipeline on 'Dataset/sample_test.csv'...")
    
    # Load sample data
    try:
        sample_df = pd.read_csv('Dataset/sample_test.csv')
    except FileNotFoundError:
        print("Error: 'Dataset/sample_test.csv' not found. Cannot run the pipeline.")
        exit()
    
    # Run the pipeline
    prediction_results = run_prediction_pipeline(sample_df)
    
    # Save the results to a CSV file
    output_path = 'Dataset/retention_candidates.csv'
    prediction_results.to_csv(output_path, index=False)

    print(f"Pipeline complete. Results saved to '{output_path}'")
    print("\n--- Sample of Results ---")
    print(prediction_results[['customerID', 'churn_prediction', 'churn_probability', 'retention_strategy']].head())
    print("\n")