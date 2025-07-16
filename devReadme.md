# Developer's Guide: Customer Churn Prediction Project

**Version: 2.0**

## 1. Project Philosophy & Core Concepts

This document is the definitive guide for developers working on this project.

The system's goal is to create an **actionable intelligence tool**. We don't just classify customers; we provide a complete workflow from data analysis to a prescriptive retention plan, served via a user-friendly interface.

Before diving into the code, make sure you understand these key concepts, as they are central to the project's design:

*   **XGBoost:** A gradient boosting framework used for our core classification model due to its high performance and the availability of tools for interpretability.
*   **SHAP (SHapley Additive exPlanations):** The foundation of our "Explainable AI" (XAI). We use the `shap.TreeExplainer` specifically for our XGBoost model. It calculates the impact of each feature on the final prediction for a *single customer*. The features with the highest absolute SHAP values are our "top churn drivers."
*   **Training-Serving Skew:** This is a critical MLOps failure mode where data used for live predictions (serving) is processed differently than the data used to train the model. We aggressively prevent this by centralizing all feature engineering logic into `data_processing.py`, which is used by all other components.
*   **FastAPI & Pydantic:** FastAPI is our choice for the backend API due to its high speed and automatic data validation/documentation. Pydantic models (e.g., `CustomerData`) are used to define the exact structure and data types of our API inputs, preventing bad data from ever reaching our model.
*   **Streamlit:** Our choice for the frontend dashboard. It allows us to build interactive data applications using pure Python, making it ideal for rapid development and iteration.

---

## 2. Data Flow Diagram

This diagram shows how data moves through the system from initial training to a final prediction in the dashboard.

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

---

## 3. Deep Dive into Code Components

### `train_model.py`
**Purpose:** To create and save all the necessary assets for our prediction model. This script is the starting point of the entire workflow.

**Key Functions & Logic:**

*   **CLV Calculation & Binning:**
    *   A simplified Customer Lifetime Value is calculated: `(MonthlyCharges * tenure) - assumed_acquisition_cost`.
    *   `pd.qcut` is used to divide the CLV of all customers in the training set into 3 quantiles (Low, Medium, High).
    *   **Crucially, the `retbins=True` argument captures the numerical boundaries of these bins.** These boundaries are saved to `clv_bins.pkl` so that we can categorize new customers using the exact same logic.

*   **Feature Engineering:**
    *   Creates interaction and ratio features (`tenure_monthly_interaction`, `tenure_monthly_ratio`) to capture non-linear relationships that the model might find useful.
    *   `premium_services_count` aggregates several binary 'Yes'/'No' columns into a single numerical feature.

*   **`objective(trial)` function:**
    *   This is the core of our **Optuna** hyperparameter search.
    *   For each `trial`, Optuna suggests a new set of hyperparameters (e.g., `trial.suggest_int('max_depth', ...)`).
    *   The function then trains an XGBoost model with these parameters using 5-fold **Stratified Cross-Validation**. Stratification ensures that each fold has the same proportion of churners and non-churners as the whole dataset, which is vital for imbalanced data.
    *   It returns the mean F1-score across the 5 folds. Optuna's goal is to maximize this score.

*   **Threshold Tuning:**
    *   A model's `.predict()` method uses a default probability threshold of 0.5, which is rarely optimal for imbalanced datasets.
    *   We use `precision_recall_curve` to get a series of precision and recall values for every possible threshold.
    *   We then calculate the F1-score for each of these thresholds and select the one that yields the highest F1-score. This `best_threshold` is used to make the final classification, balancing the need to find churners (recall) with the need to not misclassify loyal customers (precision).

### `data_processing.py`
**Purpose:** To be the single source of truth for data preparation. It prevents training-serving skew.

**Key Functions & Logic:**

*   **`prepare_data_for_prediction(df)`:**
    1.  **Validation:** It first checks if all necessary columns are present in the input DataFrame. This is a vital safeguard, especially for the batch pipeline which processes user-uploaded CSVs.
    2.  **CLV & Features:** It applies the *exact same* CLV and feature engineering formulas as `train_model.py`.
    3.  **Binning:** It uses `pd.cut` with the `clv_bins` loaded from the pickle file to categorize the CLV. This ensures a customer with a CLV of, say, $500 is treated the same way in training and prediction.
    4.  **Encoding:** It one-hot encodes categorical columns.
    5.  **Column Alignment (The Most Critical Step):** It uses `df_encoded.reindex(columns=training_columns, fill_value=0)`. This forces the DataFrame to have the exact same columns in the exact same order as the data the model was trained on. If the new data is missing a column that the model expects (e.g., `InternetService_Fiber optic`), it adds it and fills it with `0`. If the new data has a column the model has never seen, it's dropped. This is the ultimate defense against errors from data mismatches.

### `retention_strategy.py`
**Purpose:** To translate abstract model outputs (feature names) into concrete, actionable business advice.

**Key Functions & Logic:**

*   **`get_retention_strategies(customer_data, churn_drivers)`:**
    *   It uses a dictionary (`driver_function_map`) to map specific churn drivers (e.g., `PaymentMethod_Electronic check`) to dedicated handler functions (e.g., `handle_payment_method`). This is a clean, extensible design pattern.
    *   Each handler function contains the business logic for that specific driver. For example, `handle_contract` checks the customer's *actual* contract type from the `customer_data` dictionary to provide tailored advice.
    *   It includes a generic fallback strategy in case none of the top drivers have a specific handler, ensuring the user always gets some advice.

### `api/main.py`
**Purpose:** To serve the model as a high-performance, real-time web service.

**Key Functions & Logic:**

*   **Startup Logic:** The `model` and `shap.TreeExplainer` are created **once** when the application starts. Creating a SHAP explainer can be slow, so doing this at startup makes individual API calls much faster.
*   **`CustomerData` (Pydantic Model):** This class acts as a strict schema for the incoming JSON request body. If a request is missing a field or has the wrong data type (e.g., `tenure` as a string), FastAPI automatically rejects it with a clear error message before our code even runs.
*   **`@app.post("/predict")` endpoint:**
    1.  It receives the validated `customer_data`.
    2.  Converts the single customer's data into a one-row pandas DataFrame: `pd.DataFrame([customer_data.dict()])`.
    3.  Passes this DataFrame to the centralized `prepare_data_for_prediction` function.
    4.  Gets the prediction and calculates the SHAP values for the single prepared row.
    5.  Extracts the top 3 drivers by looking at the absolute SHAP values.
    6.  Returns a clean JSON response.

### `dashboard/app.py`
**Purpose:** To provide a human-friendly graphical interface for the entire system.

**Key Functions & Logic:**

*   **Batch Prediction Section:**
    *   Uses `st.file_uploader` to allow a user to select a CSV.
    *   If a file is uploaded, it calls `run_prediction_pipeline` (imported from `prediction_pipeline.py`) to process the entire file.
    *   It displays a summary of the results and provides a download button for the full output CSV.

*   **Real-time Prediction Section:**
    *   Uses `st.form` to group all the input widgets. This ensures the prediction logic only runs when the "Predict Churn" button is explicitly clicked.
    *   When submitted, it gathers all the inputs into a dictionary that matches the `CustomerData` Pydantic model.
    *   It uses the `requests` library to send this dictionary as a JSON payload to the FastAPI server's `/predict` endpoint (`requests.post(api_url, ...)`).
    *   It then receives the JSON response from the API and uses the prediction, probability, and drivers to display a formatted result (e.g., using `st.error` for a churn prediction) and the recommended retention strategies.

---

This guide should provide all the necessary details to understand, maintain, and extend the project. Welcome aboard!