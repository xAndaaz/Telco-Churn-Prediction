# Developer's Guide: ChurnAdvisor Project

**Version: 5.0**

## 1. Project Philosophy: The Advisory Approach
This document is the definitive technical guide for developers.
The system's goal is to create an **advisory intelligence tool**. We do not prescribe specific business actions. Instead, we provide a complete, data-driven workflow that generates a rich **Churn Risk Profile** for each customer.

---

## 2. Project Structure
The project follows a standard Python package structure to ensure maintainability and scalability.

```
├── churnadvisor/            # Main source code as an installable package
│   ├── __init__.py
│   ├── analysis/            # Core analysis logic (e.g., insight generation)
│   ├── pipelines/           # Scripts that orchestrate prediction steps
│   ├── processing/          # Data cleaning, preparation, and feature engineering
│   └── training/            # Model training and tuning scripts
│
├── dashboard/               # Streamlit dashboard application
├── scripts/                 # Standalone or one-off analysis scripts
│
├── Dataset/                 # Raw and processed data
├── Models/                  # Trained model artifacts
│
├── devReadme.md             # This developer guide
├── README.md                # Main project README
└── requirements.txt         # Project dependencies
```

---

## 3. Core Concepts & Key Technologies
*   **XGBRFClassifier:** Our core classification model, chosen after a data-driven benchmarking process.
*   **SMOTEENN:** A hybrid sampling technique used to address class imbalance.
*   **Context-Aware XAI:** We use `shap.TreeExplainer` to find the **top 5 SHAP drivers**, which are then fed into our custom `churnadvisor.analysis.churn_analyzer` module to generate factually consistent insights.
*   **Survival Analysis (CoxPH Model):** We use the `lifelines` library to model the *time-to-event* for churn.
*   **Absolute Paths & Imports:** All scripts within the `churnadvisor` package use absolute paths constructed from the project root and absolute package imports (e.g., `from churnadvisor.processing import ...`) to ensure they can be run and imported reliably from any context.

---

## 4. The Unified Data Flow
The entire system is orchestrated by the Streamlit dashboard, which executes the pipelines in sequence and passes dataframes directly in-memory.

```
[User Uploads CSV]
           |
           v
[dashboard/app.py] --> Calls prediction & survival pipelines in-memory
           |
           v
[churnadvisor/pipelines/*] --> Return DataFrames
           |
           v
[dashboard/app.py] --> Calls master pipeline with in-memory DataFrames
           |
           v
[dashboard/app.py] --> Loads and displays the final result DataFrame
```

---

## 5. Deep Dive into Code Components

### `churnadvisor/training/train_model.py`
**Purpose:** To perform a rigorous, data-driven process of model selection, tuning, and training.

### `churnadvisor/analysis/churn_analyzer.py`
**Purpose:** The "translation layer" between the technical model outputs and the business user. This is the core of our context-aware insight engine.

### `dashboard/app.py`
**Purpose:** The user-facing interface for the entire system.
*   **Key Logic:**
    *   **State Management:** Uses `st.session_state` for a persistent UI.
    *   **Batch Processing:** Implements a **master-detail interface**. It displays a compact, scrollable `st.dataframe` of at-risk customers (the master view). An `st.selectbox` allows the user to choose a customer, which then displays their full `ActionableInsight` in a dedicated container (the detail view).
    *   **Instant Prediction:** Provides a form for on-demand analysis of a single customer.

---
This guide should provide all the necessary details to understand, maintain, and extend the project. Welcome aboard!
