import ast

def generate_actionable_insight(customer_row):
    """
    Generates a concise, context-aware insight based on a customer's top churn drivers
    and their actual, non-encoded data.
    """
    if customer_row.get('churn_prediction', 0) == 0:
        return "No churn risk detected."

    positive_insights = []
    negative_insights = []
    
    try:
        drivers_str = customer_row.get('top_churn_drivers', '[]')
        drivers = ast.literal_eval(drivers_str) if isinstance(drivers_str, str) else drivers_str
    except (ValueError, SyntaxError):
        drivers = []

#Context-Aware Insight Mapping 
    for driver in drivers:
        if driver == 'Contract_Month-to-month':
            if customer_row.get('Contract') == 'Month-to-month':
                negative_insights.append("They are on a flexible Month-to-Month contract, a primary risk factor due to low commitment.")
            else:
                positive_insights.append("Their long-term contract is a key positive factor reducing their churn risk.")

        elif driver == 'tenure':
            tenure = customer_row.get('tenure', 0)
            if tenure <= 12:
                negative_insights.append(f"As a new customer (~{tenure:.0f} months), they are in a critical early phase and may not be fully engaged yet.")
            elif tenure > 48:
                positive_insights.append(f"Their long tenure (~{tenure:.0f} months) is a significant loyalty indicator.")

        elif driver == 'InternetService_Fiber optic':
            if customer_row.get('InternetService') == 'Fiber optic':
                negative_insights.append("The high cost of Fiber Optic internet is likely contributing to their churn risk.")
            else:
                positive_insights.append("Their choice of a non-premium internet service is a positive factor, suggesting they are not price-sensitive.")

        elif driver == 'PaymentMethod_Electronic check':
            if customer_row.get('PaymentMethod') == 'Electronic check':
                negative_insights.append("Their use of Electronic Check, correlated with less stable payment patterns, is a risk factor.")
            else:
                positive_insights.append("Their use of an automatic or stable payment method is a positive contributor.")

        elif driver == 'TechSupport_No':
            if customer_row.get('TechSupport') == 'No':
                negative_insights.append("A lack of tech support suggests they may feel unsupported or have unresolved technical issues.")
            else:
                positive_insights.append("Their subscription to Tech Support is a key protective factor.")

        elif driver == 'TotalCharges':
            if customer_row.get('TotalCharges', 0) < customer_row.get('MonthlyCharges', 0) * 6:
                negative_insights.append("Their low Total Charges suggest they are a relatively new or low-usage customer.")

        elif driver == 'tenure_monthly_ratio':
            if customer_row.get('tenure_monthly_ratio', 0) < 0.5:
                negative_insights.append("Their monthly charge is high relative to their short tenure, indicating a potential value mismatch.")

        elif driver == 'premium_services_count':
            premium_services = customer_row.get('premium_services_count', 0)
            if premium_services <= 1:
                negative_insights.append("A low number of premium services suggests a lack of deep engagement with the product ecosystem.")
            elif premium_services >= 4:
                positive_insights.append("Their high engagement with multiple premium services is a positive factor.")


    # Final Summary combining insights
    time_based_risk = customer_row.get('TimeBasedRisk', 'Medium Risk')
    final_summary = f"**Urgency:** {time_based_risk}\n\n"

    if negative_insights:
        final_summary += "**Primary Risk Factors:**\n"
        for insight in negative_insights:
            final_summary += f"- {insight}\n"
    
    if positive_insights:
        final_summary += "\n**Key Positive Factors:**\n"
        for insight in positive_insights:
            final_summary += f"- {insight}\n"

    if not negative_insights and not positive_insights and drivers:
        driver_list = ", ".join([d.replace('_', ' ') for d in drivers])
        final_summary += f"Manual review recommended. Top drivers identified: {driver_list}."
    elif not drivers:
        final_summary += "Could not determine specific drivers, manual review recommended."
        
    return final_summary.strip()


