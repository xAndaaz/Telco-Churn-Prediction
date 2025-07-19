import pandas as pd

from churn_analyzer import generate_actionable_insight

def generate_churn_risk_profiles():
    """
    Orchestrates the creation of unified churn risk profiles by combining the outputs
    of the classification and survival analysis pipelines.
    """
    print("--- Starting Unified Churn Risk Profile Generation ---")

    # 1. Load the data from the two pipelines
    try:
        classification_results = pd.read_csv('Dataset/retention_candidates.csv')
        survival_results = pd.read_csv('Dataset/survival_retention_plan.csv')
        print("Successfully loaded data from classification and survival pipelines.")
    except FileNotFoundError as e:
        print(f"Error: Could not find input file. {e}")
        print("Please ensure both 'prediction_pipeline.py' and 'survival_retention_strategy.py' have been run successfully.")
        return

    # 2. Merge the two dataframes into a single master view
    master_df = pd.merge(classification_results, survival_results, on='customerID', how='left')
    print("Successfully merged pipeline outputs.")

    # 3. Implement Quantile-Based Risk Tiering
    churn_mask = master_df['churn_prediction'] == 1
    if churn_mask.any():
        churn_probs = master_df.loc[churn_mask, 'churn_probability']
        
        high_risk_threshold = churn_probs.quantile(0.66)
        medium_risk_threshold = churn_probs.quantile(0.33)

        def assign_risk_tier(prob):
            if prob >= high_risk_threshold:
                return 'High'
            elif prob >= medium_risk_threshold:
                return 'Medium'
            else:
                return 'Low'

        master_df.loc[churn_mask, 'ProbabilityRiskTier'] = churn_probs.apply(assign_risk_tier)
    master_df['ProbabilityRiskTier'] = master_df['ProbabilityRiskTier'].fillna('Not At Risk')
    print("Successfully created quantile-based risk tiers for at-risk customers.")

    # 4. Generate Actionable Insights
    print("Generating actionable insights for each at-risk customer...")
    master_df['ActionableInsight'] = master_df.apply(generate_actionable_insight, axis=1)
    print("Insights generated.")

    # 5. Final Cleanup
    # Drop the old, now redundant, strategy columns
    final_df = master_df.drop(columns=['retention_strategy_x', 'retention_strategy_y'])
    
    # Reorder columns for better readability
    cols_to_front = ['customerID', 'churn_prediction', 'churn_probability', 'ProbabilityRiskTier', 'ActionableInsight', 'clv_tier']
    other_cols = [col for col in final_df.columns if col not in cols_to_front]
    final_df = final_df[cols_to_front + other_cols]
    
    # 6. Save the final output
    output_path = 'Dataset/master_retention_plan.csv'
    final_df.to_csv(output_path, index=False)
    print(f"\nProcess complete. Unified Churn Risk Profiles saved to '{output_path}'")
    
    print("\n--- Sample of Final Churn Risk Profiles ---")
    # Print a sample of the most important columns
    print(final_df[['customerID', 'ProbabilityRiskTier', 'ActionableInsight']].head())

if __name__ == '__main__':
    generate_churn_risk_profiles()



if __name__ == '__main__':
    generate_churn_risk_profiles()
