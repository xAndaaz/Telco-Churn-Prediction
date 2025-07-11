
"""
This module provides functions to generate actionable retention strategies
based on customer data and model-driven churn drivers.
"""

def get_retention_strategies(customer_data, churn_drivers):
    """
    Generates a list of actionable retention strategies based on churn drivers
    and the specific customer's context.

    Args:
        customer_data (dict): A dictionary containing the customer's data.
        churn_drivers (list): A list of the top churn drivers from the model.

    Returns:
        list: A list of human-readable, actionable retention strategies.
    """
    strategies = []
    processed_drivers = set()

    # --- 1. Define Strategy Functions for Different Drivers ---
    # This approach allows for more complex, conditional logic for each driver.

    def handle_contract(driver):
        """Generates strategy based on the customer's actual contract type."""
        if "Contract" in driver:
            actual_contract = customer_data.get('Contract')
            if actual_contract == 'Month-to-month':
                strategies.append(
                    "**Strategy for Contract:** This customer is on a flexible Month-to-Month plan, which makes it easy to churn. "
                    "Offer a significant discount for switching to a one-year or two-year contract to increase "
                    "their long-term commitment and provide price stability."
                )
            elif actual_contract == 'One year':
                strategies.append(
                    "**Strategy for Contract:** This customer is on a One-Year contract but is still flagged as a churn risk. "
                    "If their tenure is approaching 12, 24, or 36 months, proactively offer a renewal with a loyalty discount. "
                    "Otherwise, this may indicate dissatisfaction with the service itself."
                )
            elif actual_contract == 'Two year':
                strategies.append(
                    "**High-Priority Alert:** This customer is on a Two-Year contract, which is unusual for a churn risk. "
                    "This indicates a potentially high-value customer is deeply unhappy. A personal call from a "
                    "senior retention specialist is highly recommended to understand and resolve their issues immediately."
                )
            return True
        return False

    def handle_tenure(driver):
        """Generates strategy based on customer tenure."""
        if driver == "tenure" and customer_data.get('tenure', 100) <= 12:
            strategies.append(
                "**Strategy for New Customer:** This is a relatively new customer (<= 12 months). New customers are at a higher risk of churning. "
                "Engage them with a welcome series, offer a loyalty discount, or provide a complimentary "
                "one-month premium service to demonstrate value."
            )
            return True
        return False

    def handle_monthly_charges(driver):
        """Generates strategy for high monthly charges."""
        if driver == "MonthlyCharges":
            strategies.append(
                f"**Strategy for High Bill:** The monthly charge of ${customer_data['MonthlyCharges']:.2f} is a key factor. "
                "Review their plan for potential cost-saving opportunities, offer a promotional discount for a few months, "
                "or bundle services to provide more value at a similar price point."
            )
            return True
        return False

    def handle_internet_service(driver):
        """Generates strategy related to internet service type."""
        if "InternetService" in driver:
            if customer_data.get('InternetService') == 'Fiber optic':
                strategies.append(
                    "**Strategy for Internet Service:** The customer's Fiber Optic service is a contributing factor. While it's a premium service, "
                    "it can be associated with higher costs or perceived service issues. Proactively contact the customer to ensure "
                    "their service is performing well or offer a complimentary 'Tech Support' call."
                )
            return True
        return False
        
    def handle_payment_method(driver):
        """Generates strategy for payment method."""
        if "PaymentMethod_Electronic check" in driver:
            strategies.append(
                "**Strategy for Payment Method:** Customers paying by Electronic Check are statistically more likely to churn. "
                "Encourage a switch to a more stable payment method like automatic credit card payments by offering "
                "a small, one-time bill credit for making the change."
            )
            return True
        return False

    # --- 2. Process Drivers and Generate Strategies ---
    # A map linking raw driver names to their handling function
    driver_function_map = {
        'Contract_One year': handle_contract,
        'Contract_Two year': handle_contract,
        'tenure': handle_tenure,
        'MonthlyCharges': handle_monthly_charges,
        'InternetService_Fiber optic': handle_internet_service,
        'PaymentMethod_Electronic check': handle_payment_method,
    }

    for driver in churn_drivers:
        if driver in processed_drivers:
            continue
            
        handler = driver_function_map.get(driver)
        if handler:
            if handler(driver):
                # Mark all related drivers as processed
                if handler == handle_contract:
                    processed_drivers.add('Contract_One year')
                    processed_drivers.add('Contract_Two year')
                else:
                    processed_drivers.add(driver)

    # --- 3. Add a Default Fallback Strategy ---
    if not strategies:
        strategies.append(
            "**General Strategy:** This customer shows a general churn risk. A proactive check-in call to ensure satisfaction "
            "or a small loyalty discount could be effective."
        )
        
    return strategies
