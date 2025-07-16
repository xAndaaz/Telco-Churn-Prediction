

import pandas as pd
import pickle
from data_processing import prepare_data_for_survival

print("Starting Survival Prediction Pipeline...")

# 1. Load the Trained Survival Model
try:
    with open('Models/survival_model.pkl', 'rb') as f:
        cph_model = pickle.load(f)
    print("Survival model loaded successfully.")
except FileNotFoundError:
    print("Error: survival_model.pkl not found. Please run 'train_survival_model.py' first.")
    exit()

# 2. Load Sample Data for Prediction
# We use the same sample test file as the classification pipeline for consistency.
try:
    sample_df = pd.read_csv('Dataset/sample_test.csv')
    print(f"Loaded {len(sample_df)} records from sample_test.csv for prediction.")
except FileNotFoundError:
    print("Error: sample_test.csv not found. Please ensure it exists in the Dataset directory.")
    exit()

# 3. Prepare the Data for Survival Prediction
# We use the centralized function to ensure the same transformations are applied.
# 'is_training=False' is important, though our current logic is simple, this is good practice.
prepared_df = prepare_data_for_survival(sample_df.copy(), is_training=False)

# The model was trained on specific columns. We must ensure the prediction data has them.
# We'll get the columns from the model's internal state.
model_columns = cph_model.params_.index.tolist()
prepared_df = prepared_df.reindex(columns=model_columns, fill_value=0)

# 4. Predict Survival Functions
print("Predicting survival functions for the sample data...")
# This returns a DataFrame where rows are time points and columns are customers.
survival_functions = cph_model.predict_survival_function(prepared_df)

# 5. Extract Predictions at Specific Time Horizons
# Let's get the probability of survival at 6, 12, and 24 months.
time_points = [6, 12, 24]
predictions = {}

for time in time_points:
    # .loc[time] will give us the survival probability at that month for all customers
    # We transpose to make it a series that we can add to our results dataframe
    predictions[f'survival_prob_{time}_months'] = survival_functions.loc[time].values

predictions_df = pd.DataFrame(predictions, index=prepared_df.index)

# 6. Combine Original Data with Predictions
# Let's merge the predictions back with the original customer IDs for context.
results_df = sample_df[['customerID']].join(predictions_df)
print("Survival probabilities calculated.")

# 7. Save the Results
output_path = 'Dataset/survival_predictions.csv'
results_df.to_csv(output_path, index=False)

print(f"\nPrediction complete. Results saved to {output_path}")
print("\nFirst 5 rows of the output:")
print(results_df.head())

