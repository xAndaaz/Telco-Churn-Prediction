# ChurnAdvisor: Customer Churn Prediction & Risk Analysis

## Project Overview
**ChurnAdvisor** is an end-to-end machine learning application that identifies at-risk customers and generates a holistic **Churn Risk Profile** for each one. Instead of prescribing generic actions, this tool provides a rich, multi-faceted advisory report that empowers business users to make informed, data-driven retention decisions.

The solution uses a Telco Customer Churn dataset to train and serve a dual-model system, made available through a polished Streamlit dashboard and a high-performance FastAPI.

---

## Key Features
*   **Professional Project Structure:** All code is organized into a standard, maintainable Python package (`churnadvisor`).
*   **Dual-Model System:** Combines a high-precision **XGBRFClassifier** (to predict *if* and *why*) with a **Cox Proportional Hazards** survival model (to predict *when*).
*   **Context-Aware Insight Engine:** A sophisticated analysis module that intelligently interprets model outputs (SHAP drivers) in the context of each customer's actual data to generate factually consistent, actionable insights.
*   **Robust Interactive Dashboard:** A user-friendly Streamlit application (`ChurnAdvisor`) built with state management for a seamless experience. It features:
    *   **Instant Prediction:** Real-time analysis for a single customer.
    *   **Batch Analysis:** Bulk processing of customer lists from a CSV file.
    *   **Advanced Visualization:** An interactive SHAP beeswarm plot to visualize global feature impact.
*   **Production-Ready API:** A high-performance FastAPI endpoint serves the core classification model with low latency.

---

## How to Run the Project

### 1. Train the Models
Execute the training scripts from the project's root directory:
```bash
# Train the classification model
python -m churnadvisor.training.train_model

# Train the survival model
python -m churnadvisor.training.train_survival_model
```

### 2. Launch the API and Dashboard
*   **First, start the API in one terminal:**
    ```bash
    python api/main.py
    ```
*   **Then, start the dashboard in a new terminal:**
    ```bash
    streamlit run dashboard/app.py
    ```

---

<img width="2561" height="4547" alt="screencapture-localhost-8501-2025-07-22-00_28_56" src="https://github.com/user-attachments/assets/3bf5160c-90f4-4d4f-8f1e-03d5e646ae14" />


*For a detailed technical guide, including the project structure and data flow, please see `devReadme.md`.*
