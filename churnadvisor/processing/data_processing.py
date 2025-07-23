import pandas as pd
import pickle
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from churnadvisor.processing.feature_engineering import engineer_features

# Define the project root to construct absolute paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Load training columns and CLV bins once when the module is loaded
try:
    training_columns_path = os.path.join(PROJECT_ROOT, 'Models', 'training_columns.pkl')
    clv_bins_path = os.path.join(PROJECT_ROOT, 'Models', 'clv_bins.pkl')
    with open(training_columns_path, 'rb') as f:
        training_columns = pickle.load(f)
    with open(clv_bins_path, 'rb') as f:
        clv_bins = pickle.load(f)
except FileNotFoundError:
    training_columns = None
    clv_bins = None

def prepare_data_for_prediction(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Prepares raw customer data for prediction by applying the same transformations used in training.
    """
    if clv_bins is None or training_columns is None:
        raise RuntimeError("CLV bins or training columns not loaded. Run 'train_model.py' first.")

    df = engineer_features(df, is_training=False)
    df['clv_tier'] = pd.cut(df['clv'], bins=clv_bins, labels=['Low', 'Medium', 'High'], include_lowest=True)
    df_with_tiers = df.copy()

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df_aligned = df_encoded.reindex(columns=training_columns, fill_value=0)
    
    return df_aligned, df_with_tiers

def prepare_data_for_survival(df: pd.DataFrame, is_training=False) -> pd.DataFrame:
    """
    Prepares raw customer data for survival analysis.
    """
    df = engineer_features(df, is_training=is_training)

    if is_training:
        _, clv_bins_survival = pd.cut(df['clv'], bins=3, retbins=True)
    else:
        clv_bins_survival = [df['clv'].min(), df['clv'].quantile(0.33), df['clv'].quantile(0.66), df['clv'].max()]

    df['clv_tier'] = pd.cut(df['clv'], bins=clv_bins_survival, labels=['Low', 'Medium', 'High'], include_lowest=True)
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df_processed = df_encoded.drop(columns=['clv'], errors='ignore')
    
    if 'tenure' in df_processed.columns:
        df_processed = df_processed[df_processed['tenure'] > 0]
        
    return df_processed