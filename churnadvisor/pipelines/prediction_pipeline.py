import pandas as pd
import pickle
import shap
import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from churnadvisor.processing.data_processing import prepare_data_for_prediction
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# LOAD MODELS AND COLUMNS 
try:
    model_path = os.path.join(PROJECT_ROOT, 'Models', 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error loading model file: {e}")
    print("Please ensure you have run the training script to generate the necessary model files.")
    exit()

# CORE LOGIC FUNCTIONS

def run_prediction_pipeline(df):
    """
    Runs the full prediction and explanation pipeline on a dataframe.
    """
    # Prepare the data
    df_prepared, df_with_tiers = prepare_data_for_prediction(df.copy())
    
    # Get predictions
    predictions = model.predict(df_prepared)
    probabilities = model.predict_proba(df_prepared)[:, 1]
    
    # Get SHAP explanations
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_prepared)

    # Save SHAP values and prepared data for dashboard summary plot
    shap_values_path = os.path.join(PROJECT_ROOT, 'Models', 'shap_values.pkl')
    prepared_data_path = os.path.join(PROJECT_ROOT, 'Models', 'prepared_data.pkl')
    with open(shap_values_path, 'wb') as f:
        pickle.dump(shap_values, f)
    with open(prepared_data_path, 'wb') as f:
        pickle.dump(df_prepared, f)

    shap_df = pd.DataFrame(shap_values, columns=df_prepared.columns)
    top_features = shap_df.abs().apply(lambda x: x.nlargest(5).index.tolist(), axis=1)
    
    # Combine results
    results = df.copy()
    results['churn_prediction'] = predictions
    results['churn_probability'] = probabilities
    results['top_churn_drivers'] = top_features
    results['clv_tier'] = df_with_tiers['clv_tier']
    
    return results

# SCRIPT EXECUTION

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the churn prediction pipeline.")
    parser.add_argument('--input', type=str, default=os.path.join(PROJECT_ROOT, 'Dataset', 'sample_test.csv'),
                        help=f"Path to the input CSV file. Defaults to 'Dataset/sample_test.csv'.")
    args = parser.parse_args()

    # Ensure the input path is absolute
    input_path = args.input if os.path.isabs(args.input) else os.path.join(PROJECT_ROOT, args.input)
    
    print(f"Running prediction pipeline on '{input_path}'...")
    
    try:
        sample_df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: '{input_path}' not found. Cannot run the pipeline.")
        exit()
    
    prediction_results = run_prediction_pipeline(sample_df)
    
    output_path = os.path.join(PROJECT_ROOT, 'Dataset', 'retention_candidates.csv')
    prediction_results.to_csv(output_path, index=False)

    print(f"Pipeline complete. Results saved to '{output_path}'")
    print("\n--- Sample of Results ---")
    print(prediction_results[['customerID', 'churn_prediction', 'churn_probability', 'top_churn_drivers']].head())
    print("\n")

