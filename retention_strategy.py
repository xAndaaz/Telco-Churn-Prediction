
import pandas as pd

def get_retention_strategy(row):
    """
    Generates a retention strategy based on churn drivers and CLV.
    """
    churn_drivers = row['top_churn_drivers']
    clv_tier = row['clv_tier']
    
    # High-value customers get more personalized offers
    if clv_tier == 'High':
        base_strategy = "Offer a 15% discount on the next bill and a free premium service for 3 months."
    elif clv_tier == 'Medium':
        base_strategy = "Offer a 10% discount on the next bill."
    else:
        base_strategy = "Send a survey to understand their dissatisfaction."

    # Tailor the strategy based on specific churn drivers
    if 'tenure_monthly_ratio' in churn_drivers:
        return base_strategy + " Also, offer a longer-term contract with a lower monthly rate."
    elif 'InternetService_Fiber optic' in churn_drivers:
        return base_strategy + " Also, schedule a free technical check-up for their internet connection."
    elif 'Contract_One year' in churn_drivers or 'Contract_Two year' in churn_drivers:
        return base_strategy + " Also, offer a more flexible monthly plan."
    elif 'PaymentMethod_Electronic check' in churn_drivers:
        return base_strategy + " Also, offer a discount for switching to automatic credit card payments."
    else:
        return base_strategy

if __name__ == '__main__':
    # Load the retention candidates
    retention_df = pd.read_csv('Dataset/retention_candidates.csv')
    
    # Filter for customers predicted to churn
    churning_customers = retention_df[retention_df['churn_prediction'] == 1].copy()
    
    if churning_customers.empty:
        print("No customers predicted to churn. No retention strategies needed.")
    else:
        # Generate retention strategies
        churning_customers['retention_strategy'] = churning_customers.apply(get_retention_strategy, axis=1)
        
        # Display the retention plan
        print("Retention Plan for Churning Customers:")
        print(churning_customers[['customerID', 'clv', 'clv_tier', 'churn_probability', 'top_churn_drivers', 'retention_strategy']])
