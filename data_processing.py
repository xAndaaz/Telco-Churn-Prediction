import pandas as pd
import pickle
from feature_engineering import engineer_features

# Load training columns and CLV bins once when the module is loaded
try:
    with open('Models/training_columns.pkl', 'rb') as f:
        training_columns = pickle.load(f)
    with open('Models/clv_bins.pkl', 'rb') as f:
        clv_bins = pickle.load(f)
except FileNotFoundError:
    # This is not an error if we are just training the survival model, so we can pass.
    training_columns = None
    clv_bins = None


def prepare_data_for_prediction(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Prepares raw customer data for prediction by applying the same transformations used in training.
    This is for the XGBoost classification model.
    Returns two dataframes: one with the clv_tier for strategies, and one aligned for the model.
    """
    if clv_bins is None or training_columns is None:
        raise RuntimeError("CLV bins or training columns not loaded. Run 'train_model.py' first.")

    # Apply all feature engineering
    df = engineer_features(df, is_training=False)
    
    # Apply the pre-calculated CLV bins from training
    df['clv_tier'] = pd.cut(df['clv'], bins=clv_bins, labels=['Low', 'Medium', 'High'], include_lowest=True)
    
    # Keep a copy of the data with the human-readable clv_tier before encoding
    df_with_tiers = df.copy()

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    df_aligned = df_encoded.reindex(columns=training_columns, fill_value=0)
    
    return df_aligned, df_with_tiers

def prepare_data_for_survival(df: pd.DataFrame, is_training=False) -> pd.DataFrame:
    """
    Prepares raw customer data for survival analysis.
    This is for the Cox Proportional Hazards model.
    """
    # Apply all feature engineering
    df = engineer_features(df, is_training=is_training)

    if not is_training:
        # For prediction, you would load pre-saved bins. This is a placeholder.
        # To make this runnable, we'll just use some default bins if not training.
        # A more robust solution would be required for a true prediction pipeline.
        clv_bins_survival = [df['clv'].min(), df['clv'].quantile(0.33), df['clv'].quantile(0.66), df['clv'].max()]
    else:
        # When training, we define the bins
        _, clv_bins_survival = pd.cut(df['clv'], bins=3, retbins=True)

    df['clv_tier'] = pd.cut(df['clv'], bins=clv_bins_survival, labels=['Low', 'Medium', 'High'], include_lowest=True)
    
    # One-hot encode categorical features
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Drop the raw CLV score as it's now represented by tiers
    df_processed = df_encoded.drop(columns=['clv'], errors='ignore')
    
    # Ensure 'tenure' is greater than 0 for the model
    if 'tenure' in df_processed.columns:
        df_processed = df_processed[df_processed['tenure'] > 0]
        
    return df_processed