import pandas as pd
import pickle

# Load training columns and CLV bins once when the module is loaded
try:
    with open('Models/training_columns.pkl', 'rb') as f:
        training_columns = pickle.load(f)
    with open('Models/clv_bins.pkl', 'rb') as f:
        clv_bins = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error loading model files in data_processing.py: {e}")
    print("Please ensure 'train_model.py' has been run successfully.")
    # Propagate the error to prevent the application from running with missing files
    raise

def prepare_data_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares raw customer data for prediction by applying the same transformations used in training.

    This function centralizes the feature engineering logic. It includes validation to ensure
    that the input DataFrame has the necessary columns for processing.

    Args:
        df (pd.DataFrame): The input DataFrame with raw customer data.

    Returns:
        pd.DataFrame: A DataFrame with features engineered, encoded, and aligned with the training data.
    """
    # Ensure required columns exist before calculations
    required_cols = ['MonthlyCharges', 'tenure', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in input data: {col}")

    # Use the same CLV calculation as in training
    assumed_acquisition_cost = 100
    df['clv'] = (df['MonthlyCharges'] * df['tenure']) - assumed_acquisition_cost
    
    # Use the pre-calculated bins from training to categorize CLV
    df['clv_tier'] = pd.cut(df['clv'], bins=clv_bins, labels=['Low', 'Medium', 'High'], include_lowest=True)
    
    # Recreate the same engineered features
    df['tenure_monthly_interaction'] = df['tenure'] * df['MonthlyCharges']
    premium_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    df['premium_services_count'] = df[premium_services].apply(lambda x: (x == 'Yes').sum(), axis=1)
    df['tenure_monthly_ratio'] = df['tenure'] / (df['MonthlyCharges'] + 1e-6)
    
    # One-hot encode categorical features
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Align columns with the training data
    df_aligned = df_encoded.reindex(columns=training_columns, fill_value=0)
    
    return df_aligned
