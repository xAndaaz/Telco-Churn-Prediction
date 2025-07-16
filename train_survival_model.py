

import pandas as pd
import pickle
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split

from data_processing import prepare_data_for_survival

print("Starting Survival Analysis Model Training...")

# 1. Load Dataset
df = pd.read_csv(r'Dataset/newds.csv')

# 2. Prepare Data using Centralized Function
# This ensures consistency and adheres to our project's best practices.
df_survival = prepare_data_for_survival(df, is_training=True)

# 3. Split Data
# We split to check for consistency later, though for the first pass we'll fit on all data.
train_df, test_df = train_test_split(df_survival, test_size=0.2, stratify=df_survival['Churn'], random_state=42)

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# 5. Fit the Cox Proportional Hazards Model
# We will use the training data to fit the model.
cph = CoxPHFitter(penalizer=0.1) # A small penalizer helps with convergence and prevents overfitting

# The 'fit' method takes the dataframe, duration column, and event column.
# All other columns are treated as covariates.
print("\nFitting the Cox Proportional Hazards model...")
cph.fit(train_df, duration_col='tenure', event_col='Churn')

print("Model fitting complete. Summary of the model:")
cph.print_summary()

# 6. Evaluate the Model (Concordance Index)
# The concordance index (C-index) is a measure of rank correlation between predicted risk scores and actual time-to-event.
# It's similar to AUC for classification models. A value of 1.0 is perfect, 0.5 is random.
print("\nEvaluating model performance on the test set...")
concordance_index = cph.score(test_df, scoring_method="concordance_index")
print(f"Concordance Index on Test Data: {concordance_index:.4f}")


# 7. Save the Trained Survival Model
print("\nSaving the trained survival model...")
with open('Models/survival_model.pkl', 'wb') as f:
    pickle.dump(cph, f)

print("\nSurvival model training and saving complete.")


