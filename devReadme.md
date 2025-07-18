# Developer's Guide: Customer Churn Prediction Project

**Version: 3.0**

## 1. Project Philosophy & Core Concepts

This document is the definitive guide for developers working on this project.

The system's goal is to create an **actionable intelligence tool**. We don't just classify customers; we provide a complete workflow from data analysis to a prescriptive retention plan, served via a user-friendly interface.

Before diving into the code, make sure you understand these key concepts, as they are central to the project's design:

*   **XGBRFClassifier:** A Random Forest variant from the XGBoost library. It was chosen as our core classification model after a data-driven benchmarking process showed it had the best baseline performance. It offers a great balance of speed and predictive power.
*   **SHAP (SHapley Additive exPlanations):** The foundation of our "Explainable AI" (XAI). We use the `shap.TreeExplainer` specifically for our tree-based model. It calculates the impact of each feature on the final prediction for a *single customer*. The features with the highest absolute SHAP values are our "top churn drivers."
*   **Survival Analysis (CoxPH Model):** A more advanced statistical method that reframes the problem from *if* a customer will churn to *when* they are likely to churn. We use the `lifelines` library's `CoxPHFitter` to model churn risk over time. This provides a much richer, time-sensitive view of customer behavior.
*   **Training-Serving Skew:** This is a critical MLOps failure mode where data used for live predictions (serving) is processed differently than the data used to train the model. We aggressively prevent this by centralizing all feature engineering logic into `feature_engineering.py` and having a single `data_processing.py` module handle all data preparation.
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
   - Imports logic from [feature_engineering.py]
   - Benchmarks multiple classifiers (Decision Tree, RF, XGBoost)
   - Selects XGBRFClassifier as champion
   - Tunes Hyperparameters (Optuna)
   - Trains final model
   - SAVES --> [model.pkl, training_columns.pkl, clv_bins.pkl]
           |
           +------------------------------------------------+
           |                                                |
           v                                                v
[2. data_processing.py]                             [3. api/main.py]
   - Imports logic from [feature_engineering.py]    - Loads model.pkl & SHAP explainer at startup
   - Contains prepare_data_for_prediction()         - Exposes /predict endpoint
   - USES --> [training_columns.pkl, clv_bins.pkl]    |
           ^                                                | USES
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
   - Calls prepare_data_for_survival() from [data_processing.py]
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

### `feature_engineering.py`
**Purpose:** To be the single source of truth for creating new features. This module is imported by `data_processing.py` to ensure consistency.
*   **Key Logic:** Contains the `engineer_features` function which calculates Customer Lifetime Value (CLV) and other interaction features.

### `train_model.py`
**Purpose:** To perform a rigorous, data-driven process of model selection, tuning, and training for our **binary classification** task.
*   **Key Logic:**
    1.  **Feature Engineering:** Imports and runs `engineer_features`.
    2.  **Benchmarking:** Trains and evaluates several baseline models (Decision Tree, Random Forest, XGBoost, XGBRFClassifier) to find the best performer on default settings. The results are saved to `experiments.json`.
    3.  **Hyperparameter Tuning:** Uses the Optuna library to perform a search for the best hyperparameters for the champion model (`XGBRFClassifier`), optimizing for F1-score.
    4.  **Final Training:** Trains the `XGBRFClassifier` on the full training set using the best parameters found by Optuna.
    5.  **Threshold Tuning:** Calculates the optimal probability threshold to maximize the F1-score, balancing precision and recall.
    6.  **Serialization:** Saves the final trained model (`model.pkl`), the list of training columns (`training_columns.pkl`), and the CLV bin definitions (`clv_bins.pkl`) to the `Models/` directory.

### `data_processing.py`
**Purpose:** To be the single source of truth for all data preparation, preventing training-serving skew.
*   **`prepare_data_for_prediction(df)`:** Prepares data for the **XGBRFClassifier model**. It calls `engineer_features`, applies one-hot encoding, and aligns the columns to match the training data. It returns both the model-ready data and a version with the human-readable `clv_tier` for use in the prediction pipeline.
*   **`prepare_data_for_survival(df)`:** Prepares data for the **CoxPH survival model**. It also calls `engineer_features` but performs slightly different processing to keep the `tenure` and `Churn` columns in the format required by the `lifelines` library.

### `api/main.py`
**Purpose:** To serve the **classification model** as a high-performance, real-time web service.
*   **Key Logic:**
    *   **Optimized Loading:** The `XGBRFClassifier` model and the `shap.TreeExplainer` are loaded into memory **only once** when the FastAPI application starts. This is a critical optimization that prevents the computationally expensive explainer from being re-created on every API call, ensuring low latency.
    *   **Prediction Endpoint:** The `/predict` endpoint accepts customer data via a Pydantic model, calls the `prepare_data_for_prediction` function, and returns a JSON object with the churn prediction, probability, and a list of the top 3 churn drivers from SHAP.

---

This guide should provide all the necessary details to understand, maintain, and extend the project. Welcome aboard!