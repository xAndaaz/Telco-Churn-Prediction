# Developer's Guide: Customer Churn Prediction Project

**Version: 4.0**

## 1. Project Philosophy: The Advisory Approach

This document is the definitive technical guide for developers.

The system's goal is to create an **advisory intelligence tool**. We do not prescribe specific business actions. Instead, we provide a complete, data-driven workflow that generates a rich **Churn Risk Profile** for each customer. This profile empowers business users with a deep understanding of the *'if'*, *'why'*, and *'when'* of churn risk, allowing them to formulate the best retention strategies based on their own domain knowledge and resources.

---

## 2. Core Concepts & Key Technologies

*   **XGBRFClassifier:** A Random Forest variant from the XGBoost library. It was chosen as our core classification model after a data-driven benchmarking process showed it had the best baseline performance.
*   **SMOTEENN:** A sophisticated hybrid sampling technique used to address the class imbalance in the dataset. It combines SMOTE (over-sampling) with ENN (cleaning), which resulted in a high-precision model that is more reliable for high-cost retention scenarios.
*   **SHAP (SHapley Additive exPlanations):** The foundation of our "Explainable AI" (XAI). We use `shap.TreeExplainer` to calculate the impact of each feature on the final prediction for a single customer. These are our "top churn drivers."
*   **Survival Analysis (CoxPH Model):** A statistical method that models the *time-to-event*. We use the `lifelines` library's `CoxPHFitter` to predict *when* a customer is likely to churn, providing a time-based risk dimension.
*   **Quantile-Based Tiering:** A robust method for segmenting at-risk customers. Instead of using fixed probability thresholds, we rank churners by their probability score and group them into "High", "Medium", and "Low" risk tiers based on their percentile. This adapts to the model's probability distribution.
*   **Centralized Logic:** All feature creation is centralized in `feature_engineering.py`, and all data preparation is handled by `data_processing.py` to prevent training-serving skew.

---

## 3. The Unified Data Flow

The entire system is orchestrated by the Streamlit dashboard, which executes the pipelines in sequence.

```
[User Uploads CSV]
           |
           v
[dashboard/app.py] --> Saves to [Dataset/temp_uploaded_data.csv]
           |
           +-----------------------------------------------------------------+
           |                                                                 |
           v (Executes)                                                      v (Executes)
[1. prediction_pipeline.py]                                       [2. survival_prediction_pipeline.py]
   - Loads temp data                                                 - Loads temp data
   - Predicts churn & SHAP drivers                                   - Predicts survival probabilities
   - SAVES --> [Dataset/retention_candidates.csv]                    - SAVES --> [Dataset/survival_predictions.csv]
           |                                                                 |
           |                                                                 v (Executes)
           |                                                      [3. survival_risk_analyzer.py]
           |                                                         - Loads survival_predictions.csv
           |                                                         - Assigns time-based risk tiers
           |                                                         - SAVES --> [Dataset/survival_risk_analysis.csv]
           |                                                                 |
           +------------------------+----------------------------------------+
                                    |
                                    v (Executes)
                       [4. master_retention_pipeline.py]
                          - Loads retention_candidates.csv
                          - Loads survival_risk_analysis.csv
                          - Merges the data
                          - Applies quantile-based tiering
                          - Calls [churn_analyzer.py] to generate insights
                          - SAVES --> [Dataset/master_retention_plan.csv]
                                    |
                                    v
                       [dashboard/app.py] --> Loads and displays the final result
```

---

## 4. Deep Dive into Code Components

### `train_model.py`
**Purpose:** To perform a rigorous, data-driven process of model selection, tuning, and training.
*   **Key Logic:**
    1.  **Benchmarking:** Compares several baseline models and saves the results to `experiments.json`.
    2.  **Imbalance Handling:** Applies **SMOTEENN** to the training set to create a balanced and clean dataset for the model.
    3.  **Hyperparameter Tuning:** Uses **Optuna** to find the optimal hyperparameters for the `XGBRFClassifier`.
    4.  **Serialization:** Saves the final trained model (`model.pkl`) and other assets.

### `churn_analyzer.py`
**Purpose:** To be the "translation layer" between the technical model outputs and the business user.
*   **Key Logic:** Contains the `generate_actionable_insight` function, which uses a mapping of SHAP drivers and survival risk tiers to generate a human-readable text summary for each customer.

### `master_retention_pipeline.py`
**Purpose:** The central orchestrator that creates the final Churn Risk Profile.
*   **Key Logic:** Merges the outputs of the classification and survival pipelines, applies quantile-based risk tiering, and uses the `churn_analyzer` to create the final `ActionableInsight` column.

### `dashboard/app.py`
**Purpose:** The user-facing interface for the entire system.
*   **Key Logic:**
    *   **Batch Processing:** Provides a file uploader that triggers a `subprocess` call to run the entire sequence of pipelines, displaying the final unified result.
    *   **Real-time Analysis:** Includes a form for single-customer data entry, which calls the `/predict` endpoint on the FastAPI and displays the key churn drivers.

---

This guide should provide all the necessary details to understand, maintain, and extend the project. Welcome aboard!
