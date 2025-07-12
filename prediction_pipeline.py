import pandas as pd
import pickle
import shap
from retention_strategy import get_retention_strategies
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

# SCRIPT EXECUTION (for command-line use) 

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
    print("\n--- Sample of Results")
    print(prediction_results[['customerID', 'churn_prediction', 'churn_probability', 'retention_strategy']].head())
    print("\n")