import pandas as pd
import pickle
import shap
from data_processing import prepare_data_for_prediction

# --- 1. LOAD MODELS AND COLUMNS (Load once to be used by functions) ---
try:
    with open('Models/model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error loading model file: {e}")
    print("Please ensure you have run the `train_model.py` script to generate the necessary model files.")
    # Exit if models can't be loaded, as the script is unusable.
    exit()

# --- 2. CORE LOGIC FUNCTIONS ---

def run_prediction_pipeline(df):
    """
    Runs the full prediction and explanation pipeline on a dataframe.
    
    Args:
        df (pd.DataFrame): A dataframe with customer data.
        
    Returns:
        pd.DataFrame: The original dataframe with added columns for predictions and SHAP drivers.
    """
    
    # Prepare the data
    df_prepared, df_with_tiers = prepare_data_for_prediction(df.copy())
    
    # Get predictions
    predictions = model.predict(df_prepared)
    probabilities = model.predict_proba(df_prepared)[:, 1]
    
    # Get SHAP explanations for top drivers
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_prepared)
    shap_df = pd.DataFrame(shap_values, columns=df_prepared.columns)
    top_features = shap_df.abs().apply(lambda x: x.nlargest(5).index.tolist(), axis=1)
    
    # Combine results into a single DataFrame
    results = df.copy()
    results['churn_prediction'] = predictions
    results['churn_probability'] = probabilities
    results['top_churn_drivers'] = top_features
    # Add the clv_tier from the intermediate dataframe
    results['clv_tier'] = df_with_tiers['clv_tier']
    
    return results

# SCRIPT EXECUTION (for command-line use) 

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the churn prediction pipeline.")
    parser.add_argument('--input', type=str, default='Dataset/sample_test.csv',
                        help="Path to the input CSV file. Defaults to 'Dataset/sample_test.csv'.")
    args = parser.parse_args()

    print(f"Running prediction pipeline on '{args.input}'...")
    
    # Load sample data
    try:
        sample_df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"Error: '{args.input}' not found. Cannot run the pipeline.")
        exit()
    
    # Run the pipeline
    prediction_results = run_prediction_pipeline(sample_df)
    
    # Save the results to a CSV file
    output_path = 'Dataset/retention_candidates.csv'
    prediction_results.to_csv(output_path, index=False)

    print(f"Pipeline complete. Results saved to '{output_path}'")
    print("\n--- Sample of Results")
    print(prediction_results[['customerID', 'churn_prediction', 'churn_probability', 'top_churn_drivers']].head())
    print("\n")

