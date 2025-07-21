import pandas as pd
import pickle
import os
import sys
import argparse

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from churnadvisor.processing.data_processing import prepare_data_for_survival

# Define the project root to construct absolute paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def run_survival_prediction_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loads the survival model and runs predictions on the provided dataframe.
    """
    print("Running Survival Prediction Pipeline...")

    try:
        model_path = os.path.join(PROJECT_ROOT, 'Models', 'survival_model.pkl')
        with open(model_path, 'rb') as f:
            cph_model = pickle.load(f)
    except FileNotFoundError:
        print("Error: survival_model.pkl not found. Please run the survival training script first.")
        return None

    prepared_df = prepare_data_for_survival(df.copy(), is_training=False)
    model_columns = cph_model.params_.index.tolist()
    prepared_df = prepared_df.reindex(columns=model_columns, fill_value=0)

    print("Predicting survival functions...")
    survival_functions = cph_model.predict_survival_function(prepared_df)

    time_points = [6, 12, 24]
    predictions = {f'survival_prob_{time}_months': survival_functions.loc[time].values for time in time_points}
    predictions_df = pd.DataFrame(predictions, index=prepared_df.index)

    results_df = df[['customerID']].join(predictions_df)
    print("Survival probabilities calculated.")
    
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the survival prediction pipeline.")
    parser.add_argument('--input', type=str, default=os.path.join(PROJECT_ROOT, 'Dataset', 'sample_test.csv'),
                        help="Path to the input CSV file.")
    args = parser.parse_args()

    input_path = args.input if os.path.isabs(args.input) else os.path.join(PROJECT_ROOT, args.input)
    
    print(f"Running survival prediction pipeline on '{input_path}'...")
    
    try:
        sample_df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: '{input_path}' not found. Cannot run the pipeline.")
        exit()

    prediction_results = run_survival_prediction_pipeline(sample_df)
    
    if prediction_results is not None:
        output_path = os.path.join(PROJECT_ROOT, 'Dataset', 'survival_predictions.csv')
        prediction_results.to_csv(output_path, index=False)

        print(f"Pipeline complete. Results saved to '{output_path}'")
        print("\n--- Sample of Results ---")
        print(prediction_results.head())
        print("\n")

