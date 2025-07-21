# This module translates the model's technical output (SHAP drivers) into
# human-readable, actionable insights for business users.

def generate_actionable_insight(customer_row):
    """
    Generates a concise, context-aware insight based on a customer's top churn drivers
    and their actual data.

    Args:
        customer_row (pd.Series): A row from the master DataFrame for a single customer.

    Returns:
        str: A formatted string containing the actionable insight.
    """
    if customer_row['churn_prediction'] == 0:
        return "No churn risk detected."

    insights = []
    drivers = customer_row['top_churn_drivers']

    # --- Context-Aware Insight Mapping ---
    # We check both the driver AND the customer's actual data for that feature.

    if 'Contract_Month-to-month' in drivers and customer_row.get('Contract_Month-to-month') == 1:
        insights.append("The customer is on a flexible Month-to-Month contract, a primary driver of churn risk due to low commitment.")

    if 'tenure' in drivers and customer_row.get('tenure', 0) <= 12:
        insights.append("As a relatively new customer (<= 12 months), they are in a critical early phase and may not be fully engaged with the service yet.")
    
    if 'InternetService_Fiber optic' in drivers and customer_row.get('InternetService_Fiber optic') == 1:
        insights.append("While Fiber Optic is a premium service, its higher monthly cost is a contributing factor to their churn risk.")

    if 'PaymentMethod_Electronic check' in drivers and customer_row.get('PaymentMethod_Electronic check') == 1:
        insights.append("Their payment method, Electronic Check, is statistically correlated with a higher likelihood of churn.")

    if 'TechSupport_No' in drivers:
        if customer_row.get('TechSupport_No') == 1:
            insights.append("A lack of tech support is a key risk factor, suggesting they may feel unsupported or have unresolved technical issues.")
        else:
            # This is the "protective factor" insight
            insights.append("Their subscription to Tech Support is a significant positive factor. Without it, their churn risk would be much higher, indicating other areas of concern.")
            
    if not insights:
        return "Churn risk detected, but the primary drivers require manual review. Top drivers: " + ", ".join(drivers)

    # --- Combine Insights into a Final Summary ---
    time_based_risk = customer_row.get('TimeBasedRisk', 'Risk timeline not available.')

    final_summary = f"**Urgency:** {time_based_risk}\n\n"
    final_summary += "**Key Factors Identified:**\n"
    for insight in insights:
        final_summary += f"- {insight}\n"
        
    return final_summary.strip()