import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from churnadvisor.analysis.churn_analyzer import generate_actionable_insight

# Define the project root to construct absolute paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def generate_churn_risk_profiles(classification_results: pd.DataFrame, survival_results: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the creation of unified churn risk profiles by combining the outputs
    of the classification and survival analysis pipelines.
    
    Args:
        classification_results (pd.DataFrame): DataFrame from the prediction pipeline.
        survival_results (pd.DataFrame): DataFrame from the survival risk analysis.

    Returns:
        pd.DataFrame: The final dataframe with unified churn risk profiles.
    """
    print("--- Starting Unified Churn Risk Profile Generation ---")

    #  Merge the two dataframes
    master_df = pd.merge(classification_results, survival_results, on='customerID', how='left')
    print("Successfully merged pipeline outputs.")

    #  Implement Quantile-Based Risk Tiering as we are using RF so probablilities are narrow, but model auc is good
    churn_mask = master_df['churn_prediction'] == 1
    if churn_mask.any():
        churn_probs = master_df.loc[churn_mask, 'churn_probability']
        high_risk_threshold = churn_probs.quantile(0.66)
        medium_risk_threshold = churn_probs.quantile(0.33)

        def assign_risk_tier(prob):
            if prob >= high_risk_threshold: return 'High'
            elif prob >= medium_risk_threshold: return 'Medium'
            else: return 'Low'

        master_df.loc[churn_mask, 'ProbabilityRiskTier'] = churn_probs.apply(assign_risk_tier)
    master_df['ProbabilityRiskTier'] = master_df['ProbabilityRiskTier'].fillna('Not At Risk')
    print("Successfully created quantile-based risk tiers.")

    #  Generate Actionable Insights
    print("Generating actionable insights...")
    master_df['ActionableInsight'] = master_df.apply(generate_actionable_insight, axis=1)
    print("Insights generated.")

    #  Final Cleanup and Column Reordering
    final_df = master_df.drop(columns=['top_churn_drivers'], errors='ignore')
    cols_to_front = ['customerID', 'churn_prediction', 'ProbabilityRiskTier', 'TimeBasedRisk', 'ActionableInsight', 'clv_tier', 'churn_probability']
    other_cols = [col for col in final_df.columns if col not in cols_to_front]
    final_df = final_df[cols_to_front + other_cols]
    
    print("Process complete. Returning unified profiles.")
    return final_df

if __name__ == '__main__':
    print("Running master retention pipeline as a standalone script...")
    
    # Load the data from the two pipelines using absolute paths
    try:
        classification_results_path = os.path.join(PROJECT_ROOT, 'Dataset', 'retention_candidates.csv')
        survival_results_path = os.path.join(PROJECT_ROOT, 'Dataset', 'survival_risk_analysis.csv')
        classification_results_df = pd.read_csv(classification_results_path)
        survival_results_df = pd.read_csv(survival_results_path)

        if 'top_churn_drivers' in classification_results_df.columns:
            import ast
            classification_results_df['top_churn_drivers'] = classification_results_df['top_churn_drivers'].apply(ast.literal_eval)

        # Run the main function
        final_profiles = generate_churn_risk_profiles(classification_results_df, survival_results_df)
        
        # Save the final output
        output_path = os.path.join(PROJECT_ROOT, 'Dataset', 'master_retention_plan.csv')
        final_profiles.to_csv(output_path, index=False)
        
        print(f"\nUnified Churn Risk Profiles saved to '{output_path}'")
        print("\nSample of Final Churn Risk Profiles")
        print(final_profiles[['customerID', 'ProbabilityRiskTier', 'TimeBasedRisk', 'ActionableInsight']].head())

    except FileNotFoundError as e:
        print(f"Error: Could not find input file. {e}")
        print("Please ensure both 'prediction_pipeline.py' and 'survival_risk_analyzer.py' have been run successfully.")



