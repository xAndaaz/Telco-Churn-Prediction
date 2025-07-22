import pandas as pd
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from churnadvisor.analysis.churn_analyzer import generate_actionable_insight

# Define the project root to construct absolute paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def generate_churn_risk_profiles(classification_results: pd.DataFrame, survival_results: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the creation of unified churn risk profiles by combining the outputs
    of the classification and survival analysis pipelines.
    
    Args:
        classification_results (pd.DataFrame): DataFrame from the classification pipeline.
        survival_results (pd.DataFrame): DataFrame from the survival risk analysis.

    Returns:
        pd.DataFrame: A unified DataFrame with churn risk profiles.
    """
    print("Starting Unified Churn Risk Profile Generation.....")

    # Merge the two dataframes into a single master view
    master_df = pd.merge(classification_results, survival_results, on='customerID', how='left')
    print("Successfully merged pipeline outputs.")

    # Implement Quantile-Based Risk Tiering 
    churn_mask = master_df['churn_prediction'] == 1
    if churn_mask.any():
        churn_probs = master_df.loc[churn_mask, 'churn_probability']
        
        # Handle cases with few samples where quantiles might be the same
        high_risk_threshold = churn_probs.quantile(0.66)
        medium_risk_threshold = churn_probs.quantile(0.33)

        def assign_risk_tier(prob):
            if prob >= high_risk_threshold:
                return 'High'
            # Ensure medium threshold is not higher than high threshold for small sample sizes
            elif prob >= medium_risk_threshold and medium_risk_threshold < high_risk_threshold:
                return 'Medium'
            else:
                return 'Low'

        master_df.loc[churn_mask, 'ProbabilityRiskTier'] = churn_probs.apply(assign_risk_tier)
    
    if 'ProbabilityRiskTier' not in master_df.columns:
        master_df['ProbabilityRiskTier'] = 'Not At Risk'
    else:
        master_df['ProbabilityRiskTier'] = master_df['ProbabilityRiskTier'].fillna('Not At Risk')
        
    print("Successfully created quantile-based risk tiers for at-risk customers.")

    # 4. Generate Actionable Insights
    print("Generating actionable insights for each at-risk customer...")
    master_df['ActionableInsight'] = master_df.apply(generate_actionable_insight, axis=1)
    print("Insights generated.")

    # [CleanUP] Drop the now redundant columns
    final_df = master_df.drop(columns=['top_churn_drivers'])
    
    # Reorder columns for better readability
    cols_to_front = ['customerID', 'churn_prediction', 'ProbabilityRiskTier', 'TimeBasedRisk', 'ActionableInsight', 'clv_tier', 'churn_probability']
    other_cols = [col for col in final_df.columns if col not in cols_to_front]
    final_df = final_df[cols_to_front + other_cols]
    
    # Save the final output
    output_path = os.path.join(PROJECT_ROOT, 'Dataset', 'master_retention_plan.csv')
    final_df.to_csv(output_path, index=False)
    print(f"\nProcess complete. Unified Churn Risk Profiles saved to '{output_path}'")
    
    print("\n--- Sample of Final Churn Risk Profiles ---")
    print(final_df[['customerID', 'ProbabilityRiskTier', 'TimeBasedRisk', 'ActionableInsight']].head())
    
    return final_df


if __name__ == '__main__':
    # This block allows the script to be run standalone, preserving original behavior
    print("Running master retention pipeline as a standalone script...")
    try:
        classification_results_path = os.path.join(PROJECT_ROOT, 'Dataset', 'retention_candidates.csv')
        survival_results_path = os.path.join(PROJECT_ROOT, 'Dataset', 'survival_risk_analysis.csv')
        
        class_df = pd.read_csv(classification_results_path)
        surv_df = pd.read_csv(survival_results_path)
        
        generate_churn_risk_profiles(class_df, surv_df)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find input file. {e}")
        print("Please ensure 'prediction_pipeline.py' and 'survival_risk_analyzer.py' have been run successfully.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


