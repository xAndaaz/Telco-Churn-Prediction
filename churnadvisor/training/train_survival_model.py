

import pandas as pd
import pickle
import os
import sys
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from churnadvisor.processing.data_processing import prepare_data_for_survival

# Define the project root to construct absolute paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

print("--- Starting Survival Analysis Model Training ---")

# 1. Load and Prepare Data
df = pd.read_csv(os.path.join(PROJECT_ROOT, 'Dataset', 'newds.csv'))
df_survival = prepare_data_for_survival(df, is_training=True)

# 2. Split Data
train_df, test_df = train_test_split(df_survival, test_size=0.2, stratify=df_survival['Churn'], random_state=42)
print(f"Training data shape: {train_df.shape}")

# 3. Fit the Cox Proportional Hazards Model
cph = CoxPHFitter(penalizer=0.1)
print("\nFitting the Cox Proportional Hazards model...")
cph.fit(train_df, duration_col='tenure', event_col='Churn')
print("Model fitting complete.")
cph.print_summary()

# 4. Evaluate the Model
print("\nEvaluating model performance on the test set...")
concordance_index = cph.score(test_df, scoring_method="concordance_index")
print(f"Concordance Index on Test Data: {concordance_index:.4f}")

# 5. Save the Trained Model
print("\nSaving the trained survival model...")
model_path = os.path.join(PROJECT_ROOT, 'Models', 'survival_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(cph, f)

print("\nSurvival model training and saving complete.")


