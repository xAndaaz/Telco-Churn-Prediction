import ast

def generate_actionable_insight(customer_row):
    """
    Generates a concise, context-aware insight based on a customer's top churn drivers
    and their actual data.
    """
    if customer_row['churn_prediction'] == 0:
        return "No churn risk detected."

    positive_insights = []
    negative_insights = []
    
    try:
        drivers = ast.literal_eval(customer_row['top_churn_drivers'])
    except (ValueError, SyntaxError):
        drivers = []

    # For each driver, we check the customer's actual data to determine if it's a risk or a protective factor.

    if 'Contract_Month-to-month' in drivers:
        if customer_row.get('Contract_Month-to-month') == 1:
            negative_insights.append("They are on a flexible Month-to-Month contract, a primary risk factor due to low commitment.")
        else:
            positive_insights.append("Their long-term contract is a key positive factor reducing their churn risk.")

    if 'tenure' in drivers:
        if customer_row.get('tenure', 0) <= 12:
            negative_insights.append("As a new customer, they are in a critical early phase and may not be fully engaged yet.")
        elif customer_row.get('tenure', 0) > 48:
            positive_insights.append("Their long tenure is a significant positive factor, indicating loyalty.")

    if 'InternetService_Fiber optic' in drivers:
        if customer_row.get('InternetService_Fiber optic') == 1:
            negative_insights.append("The high cost of Fiber Optic internet is likely contributing to their churn risk.")
        else:
            positive_insights.append("Their choice of a non-premium internet service is a positive factor, suggesting they are not price-sensitive.")

    if 'PaymentMethod_Electronic check' in drivers:
        if customer_row.get('PaymentMethod_Electronic check') == 1:
            negative_insights.append("Their use of Electronic Check, correlated with less stable payment patterns, is a risk factor.")
        else:
            positive_insights.append("Their use of an automatic or stable payment method is a positive contributor.")

    if 'TechSupport_No' in drivers:
        if customer_row.get('TechSupport_No') == 1:
            negative_insights.append("A lack of tech support suggests they may feel unsupported or have unresolved technical issues.")
        else:
            positive_insights.append("Their subscription to Tech Support is a key protective factor.")

    if 'tenure_monthly_ratio' in drivers:
        if customer_row.get('tenure_monthly_ratio', 0) < 0.5:
            negative_insights.append("Their monthly charge is high relative to their short tenure, indicating a potential value mismatch.")

    if 'premium_services_count' in drivers:
        if customer_row.get('premium_services_count', 0) <= 1:
            negative_insights.append("A low number of premium services suggests a lack of deep engagement with the product ecosystem.")
        elif customer_row.get('premium_services_count', 0) >= 4:
            positive_insights.append("Their high engagement with multiple premium services is a positive factor.")

    # Combine Insights into a Final Summary
    time_based_risk = customer_row.get('TimeBasedRisk', 'Risk timeline not available.')
    final_summary = f"**Urgency:** {time_based_risk}\n\n"

    if negative_insights:
        final_summary += "**Primary Risk Factors:**\n"
        for insight in negative_insights:
            final_summary += f"- {insight}\n"
    
    if positive_insights:
        final_summary += "\n**Key Positive Factors:**\n"
        for insight in positive_insights:
            final_summary += f"- {insight}\n"

    if not negative_insights and not positive_insights:
        final_summary += f"Manual review recommended. Top drivers: {', '.join(drivers)}"
        
    return final_summary.strip()