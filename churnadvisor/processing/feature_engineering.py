import pandas as pd

def engineer_features(df: pd.DataFrame, is_training: bool = False):
    """
    Applies all feature engineering steps to the raw dataframe.
    This includes CLV calculation, interaction features, and service utilization metrics.

    Args:
        df (pd.DataFrame): The input dataframe.
        is_training (bool): If True, calculates and returns the CLV bins. 
                            Otherwise, it assumes bins will be applied later.

    Returns:
        pd.DataFrame: The dataframe with engineered features.
        list (optional): The calculated CLV bins, only returned if is_training is True.
    """
    # 1. Calculate Customer Lifetime Value (CLV)
    assumed_acquisition_cost = 100
    df['clv'] = (df['MonthlyCharges'] * df['tenure']) - assumed_acquisition_cost

    clv_bins = None
    if is_training:
        # Segment customers into value tiers based on CLV
        _, clv_bins = pd.qcut(df['clv'], q=3, labels=['Low', 'Medium', 'High'], retbins=True, duplicates='drop')
        df['clv_tier'] = pd.cut(df['clv'], bins=clv_bins, labels=['Low', 'Medium', 'High'], include_lowest=True)
    
    # 2. Advanced Feature Engineering
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})

    # Interaction features
    df['tenure_monthly_interaction'] = df['tenure'] * df['MonthlyCharges']

    # Service utilization
    premium_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    df['premium_services_count'] = df[premium_services].apply(lambda x: (x == 'Yes').sum(), axis=1)

    # Tenure to monthly charges ratio
    df['tenure_monthly_ratio'] = df['tenure'] / (df['MonthlyCharges'] + 1e-6)

    # Tenure per premium service
    df['tenure_per_premium_service'] = df['tenure'] / (df['premium_services_count'] + 1e-6)

    if is_training:
        return df, clv_bins
    else:
        return df
