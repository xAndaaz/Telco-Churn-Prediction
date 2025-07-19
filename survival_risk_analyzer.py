import pandas as pd

def generate_time_based_risk(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes survival probabilities to assign a categorical risk tier.

    Args:
        predictions_df (pd.DataFrame): DataFrame containing customer IDs and their
                                       survival probabilities at different time horizons.

    Returns:
        pd.DataFrame: The original DataFrame with an added 'TimeBasedRisk' column.
    """
    
    risk_tiers = []
    
    # Define risk thresholds
    HIGH_RISK_THRESHOLD_6_MONTHS = 0.80
    MEDIUM_RISK_THRESHOLD_12_MONTHS = 0.90
    LOW_RISK_THRESHOLD_24_MONTHS = 0.95

    for _, row in predictions_df.iterrows():
        tier = "Monitor"
        
        if row['survival_prob_6_months'] < HIGH_RISK_THRESHOLD_6_MONTHS:
            tier = "Urgent"
        elif row['survival_prob_12_months'] < MEDIUM_RISK_THRESHOLD_12_MONTHS:
            tier = "Medium Risk"
        elif row['survival_prob_24_months'] < LOW_RISK_THRESHOLD_24_MONTHS:
            tier = "Low Risk"
            
        risk_tiers.append(tier)
        
    predictions_df['TimeBasedRisk'] = risk_tiers
    return predictions_df

if __name__ == "__main__":
    print("--- Generating Time-Based Risk Analysis from Survival Predictions ---")

    try:
        predictions_data = pd.read_csv('Dataset/survival_predictions.csv')
        print(f"Loaded {len(predictions_data)} records from survival_predictions.csv.")
    except FileNotFoundError:
        print("Error: survival_predictions.csv not found. Please run 'survival_prediction_pipeline.py' first.")
        exit()

    # Generate the risk tiers
    final_risk_df = generate_time_based_risk(predictions_data)

    # Save the final analysis to a new file
    output_path = 'Dataset/survival_risk_analysis.csv'
    final_risk_df.to_csv(output_path, index=False)
    print(f"Complete risk analysis saved to {output_path}")

    print("\n--- Sample of Results ---")
    print(final_risk_df[['customerID', 'TimeBasedRisk']].head())