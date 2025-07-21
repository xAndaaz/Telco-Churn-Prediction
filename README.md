# Customer Churn Prediction & Risk Analysis

## Project Overview

This project demonstrates an end-to-end machine learning workflow for identifying at-risk customers and generating a holistic **Churn Risk Profile** for each one. Instead of prescribing specific retention strategies, this tool provides a rich, multi-faceted advisory report that empowers business users to make informed decisions.

The solution uses a Telco Customer Churn dataset to train two powerful models:
1.  An **XGBRFClassifier** to predict the *likelihood* and identify the key *drivers* of churn.
2.  A **Cox Proportional Hazards** model for survival analysis to predict *when* a customer is likely to churn.

These outputs are then unified into a single, actionable Churn Risk Profile, which is made available through an interactive Streamlit dashboard and a high-performance FastAPI.

---

## Key Features

*   **Dual-Model System:** Combines a high-precision classification model (XGBRFClassifier) with a sophisticated survival analysis model to answer not just *if* a customer will churn, but also *why* and *when*.
*   **Advanced Imbalance Handling:** Uses **SMOTEENN**, a hybrid sampling technique, to create a high-precision model that is more reliable for high-cost retention scenarios.
*   **Context-Aware Insight Engine:** A sophisticated `churn_analyzer.py` module that goes beyond basic XAI. It intelligently interprets the model's outputs in the context of each customer's actual data to generate factually consistent, actionable insights. It can even identify "protective factors" (e.g., why a customer *isn't* churning).
*   **Deep Explainability (XAI):** Integrates **SHAP** to identify the top 5 drivers of churn for each individual customer, providing clear, deep, and actionable reasons for the model's predictions.
*   **Unified Churn Risk Profile:** A master pipeline that synthesizes all model outputs (churn probability, SHAP drivers, time-based risk, CLV) into a single, human-readable advisory report.
*   **Interactive Dashboard:** A user-friendly Streamlit application for both real-time single-customer analysis and batch processing of entire customer lists, complete with a **SHAP Summary Plot** to visualize global feature importance.
*   **Production-Ready API:** A high-performance FastAPI endpoint serves the core classification model, with critical components like the SHAP explainer pre-loaded for low-latency responses.

---

## The Unified Workflow

The project's core is the `master_retention_pipeline.py`, which orchestrates the following steps:

1.  **Classification Analysis (`prediction_pipeline.py`):** Predicts churn probability and SHAP drivers.
2.  **Survival Analysis (`survival_risk_analyzer.py`):** Assigns a categorical, time-based risk tier ("Urgent", "Medium Risk", etc.).
3.  **Churn Risk Profile Generation (`master_retention_pipeline.py`):** Merges the outputs and uses the `churn_analyzer.py` module to translate the technical model outputs into a human-readable **Actionable Insight**.

---

## How to Run the Project

### 1. Train the Models
First, you need to train the classification and survival models.
```bash
# Train the classification model (XGBRFClassifier)
python train_model.py

# Train the survival model (CoxPH)
python train_survival_model.py
```

### 2. Launch the API and Dashboard
To interact with the system, you need to run both the FastAPI server and the Streamlit dashboard.

*   **First, start the API in one terminal:**
    ```bash
    python api/main.py
    ```
*   **Then, start the dashboard in a new terminal:**
    ```bash
    streamlit run dashboard/app.py
    ```

### 3. Use the Dashboard
*   **Real-time Analysis:** Fill in a customer's details to get an instant Churn Risk Profile.
*   **Batch Analysis:** Upload a CSV of customers and click "Generate Churn Risk Profiles" to run the entire unified pipeline and download the results.

---
*For a more detailed technical guide, including data flow diagrams and code component deep dives, please see `devReadme.md`.*
