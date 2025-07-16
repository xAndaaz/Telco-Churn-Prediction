# Developer's Guide: Customer Churn Prediction Project

**Version: 2.1**

## 1. Project Philosophy & Core Concepts

This document is the definitive guide for developers working on this project.

The system's goal is to create an **actionable intelligence tool**. We don't just classify customers; we provide a complete workflow from data analysis to a prescriptive retention plan, served via a user-friendly interface.

Before diving into the code, make sure you understand these key concepts, as they are central to the project's design:

*   **XGBoost:** A gradient boosting framework used for our core classification model due to its high performance and the availability of tools for interpretability.
*   **SHAP (SHapley Additive exPlanations):** The foundation of our "Explainable AI" (XAI). We use the `shap.TreeExplainer` specifically for our XGBoost model. It calculates the impact of each feature on the final prediction for a *single customer*. The features with the highest absolute SHAP values are our "top churn drivers."
*   **Survival Analysis (CoxPH Model):** A more advanced statistical method that reframes the problem from *if* a customer will churn to *when* they are likely to churn. We use the `lifelines` library's `CoxPHFitter` to model churn risk over time. This provides a much richer, time-sensitive view of customer behavior.
*   **Training-Serving Skew:** This is a critical MLOps failure mode where data used for live predictions (serving) is processed differently than the data used to train the model. We aggressively prevent this by centralizing all feature engineering logic into `data_processing.py`, which is used by all other components.
*   **FastAPI & Pydantic:** FastAPI is our choice for the backend API due to its high speed and automatic data validation/documentation. Pydantic models (e.g., `CustomerData`) are used to define the exact structure and data types of our API inputs, preventing bad data from ever reaching our model.
*   **Streamlit:** Our choice for the frontend dashboard. It allows us to build interactive data applications using pure Python, making it ideal for rapid development and iteration.

---

## 2. Data Flow Diagrams

### 2.1. Classification Workflow

This diagram shows how data moves through the system for the binary churn classification task.

```
[Telco-Customer-Churn.csv]
           |
           v
[1. train_model.py]
   - Performs Feature Engineering (CLV, etc.)
   - Tunes Hyperparameters (Optuna)
   - Trains the model
   - SAVES --> [model.pkl, training_columns.pkl, clv_bins.pkl]
           |
           +------------------------------------------------+
           |                                                |
           v                                                v
[2. data_processing.py]                             [3. api/main.py]
   - Contains prepare_data_for_prediction()         - Loads model.pkl at startup
   - USES --> [training_columns.pkl, clv_bins.pkl]    - Exposes /predict endpoint
           ^                                                |
           |                                                | USES
           | IMPORTS                                        |
           |                                                v
[4. prediction_pipeline.py & dashboard/app.py (Batch)]      |
   - Takes user CSV                                         |
   - Calls prepare_data_for_prediction()                    |
   - Makes batch predictions                              [5. dashboard/app.py (Real-time)]
   - Generates retention strategies                       - Takes user form input
                                                            - Calls /predict endpoint on api/main.py
                                                            - Displays results
```

### 2.2. Survival Analysis Workflow

This diagram shows the flow for the more advanced time-to-event analysis.

```
[Telco-Customer-Churn.csv]
           |
           v
[1. train_survival_model.py]
   - Calls prepare_data_for_survival() from data_processing.py
   - Trains the CoxPH model
   - SAVES --> [survival_model.pkl]
           |
           v
[2. survival_prediction_pipeline.py]
   - Loads survival_model.pkl
   - Takes new customer data
   - Calls prepare_data_for_survival()
   - Predicts survival probabilities over time
   - SAVES --> [survival_predictions.csv]
           |
           v
[3. survival_retention_strategy.py]
    - Loads survival_predictions.csv
    - Applies time-based rules
    - Generates tiered retention strategies (Urgent, Medium Risk, etc.)
    - SAVES --> [survival_retention_plan.csv]
```

---

## 3. Deep Dive into Code Components

### `train_model.py`
**Purpose:** To create and save all the necessary assets for our **binary classification** model.

*   **Key Logic:** See original description. Focuses on XGBoost, Optuna, and F1-score optimization.

### `train_survival_model.py`
**Purpose:** To create and save the assets for our **survival analysis** model.

*   **Key Logic:**
    *   Calls the centralized `prepare_data_for_survival` function.
    *   Uses the `lifelines` library to instantiate a `CoxPHFitter` model.
    *   Fits the model on the training data, using `tenure` as the duration and `Churn` as the event.
    *   Evaluates the model using the **Concordance Index (C-index)**, which measures the model's ability to correctly rank customers by their risk.
    *   Saves the fitted model to `survival_model.pkl`.

### `data_processing.py`
**Purpose:** To be the single source of truth for data preparation. It prevents training-serving skew.

*   **`prepare_data_for_prediction(df)`:** Prepares data for the **XGBoost classification model**. See original description.
*   **`prepare_data_for_survival(df)`:** Prepares data for the **CoxPH survival model**. It performs the same feature engineering but keeps the `tenure` and `Churn` columns in their original format as required by the `lifelines` library.

### `retention_strategy.py`
**Purpose:** To translate **classification model outputs** (feature names) into concrete, actionable business advice.

*   **Key Logic:** See original description. Maps SHAP drivers to specific actions.

### `survival_retention_strategy.py`
**Purpose:** To translate **survival model outputs** (probabilities over time) into a tiered, time-sensitive retention plan.

*   **Key Logic:**
    *   Loads the `survival_predictions.csv` file.
    *   Defines risk thresholds based on survival probability at different time horizons (e.g., `<80%` survival at 6 months is "High Risk").
    *   Assigns a specific, tiered retention strategy based on the urgency of the churn risk.

### `api/main.py`
**Purpose:** To serve the **classification model** as a high-performance, real-time web service.

*   **Key Logic:** See original description.

### `dashboard/app.py`
**Purpose:** To provide a human-friendly graphical interface for the entire system.

*   **Key Logic:** See original description.

---

This guide should provide all the necessary details to understand, maintain, and extend the project. Welcome aboard!
