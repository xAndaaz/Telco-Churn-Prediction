# Customer Churn Prediction and Retention Strategy

## Project Overview

This project demonstrates an end-to-end machine learning workflow for predicting customer churn and generating proactive retention strategies. The primary goal is to identify customers who are likely to churn and provide them with targeted interventions to improve retention.

The solution uses a Telco Customer Churn dataset to train an XGBoost model. It then leverages SHAP (SHapley Additive exPlanations) to understand the key drivers behind each churn prediction. Based on these drivers and the calculated Customer Lifetime Value (CLV), a set of personalized retention strategies is automatically generated for at-risk customers.

---

## Key Features

*   **Exploratory Data Analysis (EDA):** In-depth analysis of the dataset to understand feature distributions and correlations.
*   **Advanced Feature Engineering:** Creation of new features, including Customer Lifetime Value (CLV), to improve model performance.
*   **Hyperparameter Tuning:** Use of Optuna for automated hyperparameter optimization of the XGBoost model.
*   **Predictive Modeling:** An XGBoost classifier trained to predict customer churn with high accuracy.
*   **Explainable AI (XAI):** Integration of SHAP to explain individual predictions and identify the top factors influencing churn for each customer.
*   **Automated Retention Strategy:** A rule-based system that generates personalized retention offers based on a customer's CLV and their specific churn drivers.

---

## Model Selection and Justification

To ensure the most effective model was chosen, a rigorous selection process was undertaken. Rather than defaulting to a single model, we benchmarked several powerful classifiers to compare their baseline performance on the dataset.

1.  **Benchmarking:** We first trained and evaluated the following models with their default parameters:
    *   Decision Tree
    *   Random Forest
    *   XGBoost (Random Forest variant)
    *   XGBoost (Gradient Boosted Trees)

2.  **Analysis:** The results, logged in `experiments.json`, showed that the **XGBoost RF Classifier (`XGBRFClassifier`)** provided the best balance of F1-score and AUC without any tuning.

3.  **Selection and Tuning:** Based on this evidence, the `XGBRFClassifier` was selected as the champion model for this project. We then dedicated our efforts to fine-tuning its hyperparameters using Optuna to maximize its predictive power, leading to the final model used in this pipeline. This data-driven approach ensures that our final model is not just a good choice, but a justified one.

---

## The Workflow

The project is structured as a sequential pipeline:

1.  **Data Analysis (`telco_churn_analysis.py`):**
    *   Loads the raw Telco Customer Churn data.
    *   Performs data cleaning, visualization, and initial analysis.
    *   Saves a cleaned dataset (`newds.csv`) for the next stage.

2.  **Model Training (`train_model.py`):**
    *   Loads the cleaned data.
    *   Engineers features, including a simplified Customer Lifetime Value (CLV).
    *   Uses Optuna to find the best hyperparameters for an XGBoost model.
    *   Trains the final model on the full training set and saves it (`model.pkl`).
    *   Saves the list of features used during training (`training_columns.pkl`).

3.  **Prediction and Explanation (`prediction_pipeline.py`):**
    *   Loads the trained model and a sample of test data (`sample_test.csv`).
    *   Applies the same feature engineering steps to the test data.
    *   Predicts churn probability for each customer.
    *   Uses SHAP to determine the top 3 churn drivers for each prediction.
    *   Saves the results, including predictions and drivers, to `retention_candidates.csv`.

4.  **Retention Strategy Generation (`retention_strategy.py`):**
    *   Loads the `retention_candidates.csv` file.
    *   Filters for customers predicted to churn.
    *   Applies a set of rules to generate a personalized retention strategy for each churning customer based on their CLV tier and top churn drivers.
    *   Prints the final retention plan to the console.

---

## Getting Started

### Prerequisites

*   Python 3.8+
*   pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd Churn-Predicition
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Architecture and Design

This project is designed with a clear separation of concerns, following modern MLOps best practices:

*   **Centralized Data Processing:** The feature engineering and data preparation logic is centralized in `data_processing.py`. This ensures that the exact same transformations are applied to data during training, batch prediction, and real-time API inference, preventing training-serving skew.
*   **Standalone API:** The `api/main.py` file provides a production-ready, standalone FastAPI service. It is decoupled from the training and batch-processing scripts. For efficiency, the SHAP explainer is loaded once at startup to minimize latency for individual prediction requests.
*   **Interactive Dashboard:** The `dashboard/app.py` acts as the user-facing component. It communicates with the backend API for real-time predictions and can also run the batch prediction pipeline directly, offering a flexible user experience.

---

## How to Run the Project

Execute the scripts in the following order to run the complete pipeline:

1.  **Train the Model:**
    *   This script will perform feature engineering and train the XGBoost model. The trained model and columns will be saved to the `Models/` directory.
    ```bash
    python train_model.py
    ```

2.  **Run the Prediction Pipeline (Optional):**
    *   This script will use the trained model to predict churn on the sample data and generate SHAP explanations. The output will be saved to `Dataset/retention_candidates.csv`.
    ```bash
    python prediction_pipeline.py
    ```

3.  **Launch the API and Dashboard:**
    *   To interact with the model, you need to run both the FastAPI server and the Streamlit dashboard.
    *   **First, start the API:**
        ```bash
        python api/main.py
        ```
    *   **Then, in a new terminal, start the dashboard:**
        ```bash
        streamlit run dashboard/app.py
        ```

---

## File Descriptions

*   `.gitignore`: Specifies files for Git to ignore.
*   `requirements.txt`: A list of the Python packages required to run the project.
*   `telco_churn_analysis.py`: Script for initial exploratory data analysis.
*   `train_model.py`: Script for feature engineering, model training, and hyperparameter tuning.
*   `prediction_pipeline.py`: Script to make predictions and generate SHAP explanations on new data. It imports feature engineering logic from `data_processing.py`.
*   `retention_strategy.py`: Script to create personalized retention strategies for at-risk customers.
*   `data_processing.py`: A centralized module containing the data preparation and feature engineering logic, used by both the API and the prediction pipeline to ensure consistency.
*   `api/main.py`: A FastAPI application that serves the churn prediction model. It provides a `/predict` endpoint that takes customer data and returns a prediction, probability, and key churn drivers.
*   `dashboard/app.py`: A Streamlit web application that provides a user-friendly interface for both real-time single predictions and batch predictions on CSV files.
*   `Dataset/`: Directory containing the raw, cleaned, and output data files.
*   `Models/`: Directory where the trained model (`model.pkl`) and other assets like training columns are stored.

---

## Technologies Used

*   **Python**
*   **Pandas:** For data manipulation and analysis.
*   **Scikit-learn:** For data preprocessing and model evaluation.
*   **XGBoost:** For building the high-performance gradient boosting model.
*   **Optuna:** For efficient hyperparameter optimization.
*   **SHAP:** For explaining the output of the machine learning model.
*   **FastAPI:** For building the high-performance, production-ready API.
*   **Streamlit:** For creating the interactive web dashboard.
*   **Matplotlib & Seaborn:** For data visualization.
