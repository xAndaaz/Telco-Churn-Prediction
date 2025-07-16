import pandas as pd
import pickle
from data_processing import prepare_data_for_survival

def run_survival_prediction_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loads the survival model and runs predictions on the provided dataframe.
    
    Returns:
        pd.DataFrame: A dataframe with customerID and survival probabilities.
    """
    print("Running Survival Prediction Pipeline...")

    # 1. Load the Trained Survival Model
    try:
        with open('Models/survival_model.pkl', 'rb') as f:
            cph_model = pickle.load(f)
    except FileNotFoundError:
        print("Error: survival_model.pkl not found. Please run 'train_survival_model.py' first.")
        return None

    # 2. Prepare the Data for Survival Prediction
    prepared_df = prepare_data_for_survival(df.copy(), is_training=False)
    model_columns = cph_model.params_.index.tolist()
    prepared_df = prepared_df.reindex(columns=model_columns, fill_value=0)

    # 3. Predict Survival Functions
    print("Predicting survival functions...")
    survival_functions = cph_model.predict_survival_function(prepared_df)

    # 4. Extract Predictions at Specific Time Horizons
    time_points = [6, 12, 24]
    predictions = {}
    for time in time_points:
        predictions[f'survival_prob_{time}_months'] = survival_functions.loc[time].values
    predictions_df = pd.DataFrame(predictions, index=prepared_df.index)

    # 5. Combine Original Data with Predictions
    results_df = df[['customerID']].join(predictions_df)
    print("Survival probabilities calculated.")
    
    return results_df

if __name__ == "__main__":
    print("Running survival prediction pipeline on 'Dataset/sample_test.csv'...")
    
    try:
        sample_df = pd.read_csv('Dataset/sample_test.csv')
    except FileNotFoundError:
        print("Error: 'Dataset/sample_test.csv' not found. Cannot run the pipeline.")
        exit()

    # Run the pipeline
    prediction_results = run_survival_prediction_pipeline(sample_df)
    
    if prediction_results is not None:
        # Save the results to a CSV file
        output_path = 'Dataset/survival_predictions.csv'
        prediction_results.to_csv(output_path, index=False)

        print(f"Pipeline complete. Results saved to '{output_path}'")
        print("\n--- Sample of Results ---")
        print(prediction_results.head())
        print("\n")
