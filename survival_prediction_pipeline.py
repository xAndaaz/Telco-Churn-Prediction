import pandas as pd
import pickle
from data_processing import prepare_data_for_survival

def run_survival_prediction_pipeline():
    """
    Loads the survival model and runs predictions on the sample test data.
    Saves the results to 'survival_predictions.csv'.
    """
    print("Starting Survival Prediction Pipeline...")

    # 1. Load the Trained Survival Model
    try:
        with open('Models/survival_model.pkl', 'rb') as f:
            cph_model = pickle.load(f)
        print("Survival model loaded successfully.")
    except FileNotFoundError:
        print("Error: survival_model.pkl not found. Please run 'train_survival_model.py' first.")
        return None

    # 2. Load Sample Data for Prediction
    try:
        sample_df = pd.read_csv('Dataset/sample_test.csv')
        print(f"Loaded {len(sample_df)} records from sample_test.csv for prediction.")
    except FileNotFoundError:
        print("Error: sample_test.csv not found. Please ensure it exists in the Dataset directory.")
        return None

    # 3. Prepare the Data for Survival Prediction
    prepared_df = prepare_data_for_survival(sample_df.copy(), is_training=False)
    model_columns = cph_model.params_.index.tolist()
    prepared_df = prepared_df.reindex(columns=model_columns, fill_value=0)

    # 4. Predict Survival Functions
    print("Predicting survival functions for the sample data...")
    survival_functions = cph_model.predict_survival_function(prepared_df)

    # 5. Extract Predictions at Specific Time Horizons
    time_points = [6, 12, 24]
    predictions = {}
    for time in time_points:
        predictions[f'survival_prob_{time}_months'] = survival_functions.loc[time].values
    predictions_df = pd.DataFrame(predictions, index=prepared_df.index)

    # 6. Combine Original Data with Predictions
    results_df = sample_df[['customerID']].join(predictions_df)
    print("Survival probabilities calculated.")

    # 7. Save the Results
    output_path = 'Dataset/survival_predictions.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nPrediction complete. Results saved to {output_path}")
    
    return output_path

if __name__ == "__main__":
    run_survival_prediction_pipeline()
    # The original script printed the head, we can omit that for the main execution
    results = pd.read_csv('Dataset/survival_predictions.csv')
    print("\nFirst 5 rows of the output:")
    print(results.head())