

import pandas as pd
import pickle
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split

print("Starting Survival Analysis Model Training...")

# 1. Load Dataset
df = pd.read_csv(r'Dataset/newds.csv')

# 2. Replicate Feature Engineering from the Classification Model
# This ensures that the features we test have the same basis as our original model.
# We will refine which features to keep for the survival model later.

# CLV Calculation
assumed_acquisition_cost = 100
df['clv'] = (df['MonthlyCharges'] * df['tenure']) - assumed_acquisition_cost
_, clv_bins = pd.qcut(df['clv'], q=3, labels=['Low', 'Medium', 'High'], retbins=True, duplicates='drop')
df['clv_tier'] = pd.cut(df['clv'], bins=clv_bins, labels=['Low', 'Medium', 'High'], include_lowest=True)

# Map Churn to a binary indicator (1 for event, 0 for no event)
df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})

# Interaction and ratio features
df['tenure_monthly_interaction'] = df['tenure'] * df['MonthlyCharges']
premium_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
df['premium_services_count'] = df[premium_services].apply(lambda x: (x == 'Yes').sum(), axis=1)
df['tenure_monthly_ratio'] = df['tenure'] / (df['MonthlyCharges'] + 1e-6)
df['tenure_per_premium_service'] = df['tenure'] / (df['premium_services_count'] + 1e-6)

# One-hot encode categorical features
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 3. Prepare Data for Survival Analysis
# The CoxPH model requires a dataframe with duration, event observation, and covariates.
# 'tenure' is our duration.
# 'Churn' is our event indicator.

# The lifelines library handles categorical data well, so we can start with the encoded data.
# We need to select the columns for the model. Let's exclude identifiers and the raw CLV score.
df_survival = df_encoded.drop(columns=['clv'], errors='ignore')

# Ensure 'tenure' is greater than 0, as lifelines requires positive duration
df_survival = df_survival[df_survival['tenure'] > 0]

# 4. Split Data
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


