# This module translates the model's technical output (SHAP drivers) into
# human-readable, actionable insights for business users.

def generate_actionable_insight(customer_row):
    """
    Generates a concise, human-readable insight based on a customer's top churn drivers.

    Args:
        customer_row (pd.Series): A row from the master DataFrame for a single customer.

    Returns:
        str: A formatted string containing the actionable insight.
    """
    # Only generate insights for customers predicted to churn
    if customer_row['churn_prediction'] == 0:
        return "No churn risk detected."

    insights = []
    drivers = customer_row['top_churn_drivers']

    # --- Insight Mapping ---
    # This is a simple rule-based system. More complex NLP could be used in the future.

    if 'Contract_Month-to-month' in drivers:
        insights.append("The customer is on a flexible Month-to-Month contract, which is a primary driver of their churn risk due to low commitment.")

    if 'tenure' in drivers and customer_row['tenure'] <= 12:
        insights.append("As a relatively new customer (<= 12 months), they are in a critical early phase and may not be fully engaged with the service yet.")
    
    if 'InternetService_Fiber optic' in drivers:
        insights.append("While Fiber Optic is a premium service, its higher monthly cost is a contributing factor to their churn risk.")

    if 'PaymentMethod_Electronic check' in drivers:
        insights.append("Their payment method, Electronic Check, is statistically correlated with a higher likelihood of churn.")

    if 'TechSupport_No' in drivers:
        insights.append("A lack of tech support may indicate unresolved technical issues or a perception of poor service quality.")
        
    if not insights:
        return "Churn risk detected, but the primary drivers are not covered by standard insights. A manual review is recommended."

    # --- Combine Insights into a Final Summary ---
    # We also add the time-based urgency from the survival model.
    
    # Rename the survival strategy column for clarity
    time_based_risk = customer_row.get('TimeBasedRisk', 'Risk timeline not available.')

    final_summary = f"**Urgency:** {time_based_risk}\n\n"
    final_summary += "**Key Factors Identified:**\n"
    for insight in insights:
        final_summary += f"- {insight}\n"
        
    return final_summary.strip()
