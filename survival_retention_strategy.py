

import pandas as pd

def generate_survival_retention_strategies(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates retention strategies based on customer survival probabilities.

    Args:
        predictions_df (pd.DataFrame): DataFrame containing customer IDs and their
                                       survival probabilities at different time horizons.

    Returns:
        pd.DataFrame: The original DataFrame with an added 'retention_strategy' column.
    """
    
    strategies = []
    
    # Define risk thresholds (these can be tuned based on business needs)
    HIGH_RISK_THRESHOLD_6_MONTHS = 0.80  # If survival prob at 6 months is less than 80%
    MEDIUM_RISK_THRESHOLD_12_MONTHS = 0.85 # If survival prob at 12 months is less than 85%

    for index, row in predictions_df.iterrows():
        strategy = "No Immediate Action Required (Monitor)"
        
        # Evaluate high-risk customers first, as they are the most urgent
        if row['survival_prob_6_months'] < HIGH_RISK_THRESHOLD_6_MONTHS:
            strategy = (
                "URGENT - High Risk: Customer has a >20% chance of churning within 6 months. "
                "Recommendation: Proactive wellness call from a senior support agent and a high-value, "
                "long-term contract offer (e.g., 2-year plan with 15% discount)."
            )
        # Evaluate medium-risk customers
        elif row['survival_prob_12_months'] < MEDIUM_RISK_THRESHOLD_12_MONTHS:
            strategy = (
                "Medium Risk: Customer likely to churn within the year. "
                "Recommendation: Add to a targeted email campaign offering a moderate incentive, "
                "such as a free premium service (e.g., OnlineBackup) for 6 months."
            )
            
        strategies.append(strategy)
        
    predictions_df['retention_strategy'] = strategies
    return predictions_df

if __name__ == "__main__":
    print("Generating Retention Strategies from Survival Predictions...")

    try:
        predictions_data = pd.read_csv('Dataset/survival_predictions.csv')
        print(f"Loaded {len(predictions_data)} records from survival_predictions.csv.")
    except FileNotFoundError:
        print("Error: survival_predictions.csv not found. Please run 'survival_prediction_pipeline.py' first.")
        exit()

    # Generate the strategies
    final_strategies_df = generate_survival_retention_strategies(predictions_data)

    print("\nGenerated Retention Strategies:")
    # Print the results to the console for review
    for index, row in final_strategies_df.iterrows():
        print(f"CustomerID: {row['customerID']}")
        print(f"  - Strategy: {row['retention_strategy']}\n")

    # Save the final strategies to a new file
    output_path = 'Dataset/survival_retention_plan.csv'
    final_strategies_df.to_csv(output_path, index=False)
    print(f"Complete retention plan saved to {output_path}")

