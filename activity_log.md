# Activity Log & Gemini's Thought Process

This file logs the steps taken to improve the churn prediction model, the reasoning behind those steps, and the outcomes.

## 2025-07-14

### Initial Analysis & Planning

1.  **Project Review:** Accessed and analyzed the core Python files (`data_processing.py`, `train_model.py`, `prediction_pipeline.py`, `retention_strategy.py`, `api/main.py`, etc.).
2.  **Assessment:**
    *   **Strengths:** Solid foundation with XGBoost, Optuna, SHAP, and a modular structure including a FastAPI.
    *   **Weaknesses:** Feature engineering could be more advanced, model evaluation is static, retention strategies are rule-based, and the project lacks robust MLOps practices (config management, testing, experiment tracking).
3.  **Proposed Improvement Plan:**
    *   Enhance feature engineering in `train_model.py`.
    *   Improve model evaluation and activate hyperparameter tuning.
    *   Refine retention strategies to be more data-driven.
    *   Introduce production-readiness features: a `config.yaml`, unit tests, and MLflow integration.

### Execution Log

*   **User Instruction:** User requested a more granular, step-by-step approach with approval before each change.
*   **User Instruction:** User requested that all significant changes be committed to Git.
*   **Action:** Checked Git status, added `activity_log.md`, and committed it to establish a baseline.
*   **Feature Proposal:** Proposed three new features (`tenure_per_premium_service`, `has_dependents_or_partner`, `digital_engagement_score`) in a table format.
*   **User Feedback:** User noted that the table format was unreadable in the CLI and requested a linear list format.
*   **Feature Approval:** User approved adding **only** the `tenure_per_premium_service` feature and rejected the other two.
*   **Next Action:** Update the `train_model.py` and `data_processing.py` scripts to include the new feature.
*   **Action:** Added the `tenure_per_premium_service` feature to both scripts and committed the changes.
*   **Action:** Retrained the model by executing the `train_model.py` script within the virtual environment.
*   **Observation:** The model trained successfully, but the lack of historical metrics makes it difficult to assess the impact of the new feature. This reinforces the need for experiment tracking.
*   **Next Step Proposal:** Proposed integrating MLflow and activating Optuna.
*   **User Decision:** User rejected MLflow integration but approved activating Optuna for hyperparameter tuning.
*   **Next Action:** Modify `train_model.py` to activate Optuna and comment out the static `best_params`.
*   **Action:** Activated Optuna and retrained the model to find new optimal hyperparameters.
*   **Next Step Proposal:** Refine retention strategies to be value-based using the `clv_tier`.
*   **User Decision:** User approved the plan, noting this was the original intent for the `clv_tier` feature.
*   **Bug Identified:** Discovered that the `clv_tier` is not being passed to the `get_retention_strategies` function in the prediction pipeline.
*   **Current Action:** Plan to fix the bug in `prediction_pipeline.py` before implementing the new strategy logic.
*   **Action:** Fixed the bug and implemented value-based retention strategies in `retention_strategy.py`.
*   **Next Step Proposal:** Improve project structure by adding a `config.yaml` and unit tests.
*   **User Decision:** User has decided to skip the project structure improvements.

---

## 2025-07-15

### Project Reframing & Strategic Goals

*   **User Instruction:** The user has clarified the project's business context. This is an internship project being evaluated by their company.
*   **Core Objective:** The primary goal is to demonstrate advanced data science skills to secure a positive evaluation.
*   **Key Evaluation Criteria:**
    1.  **Adherence to Work:** Following instructions and best practices.
    2.  **Problem-Solving Ability:** Effectively diagnosing and solving complex problems.
    3.  **Quality of Work:** Producing robust, reliable, and well-documented results.
    4.  **Creativity and Innovation:** Applying novel techniques and thinking beyond the initial prompt.
*   **Guideline Recap:** The initial guideline was to predict churn to enable proactive retention. Our goal is to significantly exceed this baseline.
*   **Our Strategy:** All subsequent improvements will be framed to explicitly target one or more of these evaluation criteria. We will focus on identifying and closing the gaps between a standard academic project and a professional, production-ready data science solution.
*   **User Feedback:** User requested to avoid table formats in responses as they are difficult to read in the CLI. Will use list formats going forward.

### Survival Analysis Implementation

*   **Decision:** We have decided to implement Survival Analysis to address the "Creativity & Innovation" criterion. This reframes the problem from *if* a customer will churn to *when*.
*   **Plan:**
    1.  Add `lifelines` to `requirements.txt`.
    2.  Create a new script, `train_survival_model.py`, to avoid disrupting the current pipeline.
    3.  Use the Cox Proportional Hazards model (`CoxPHFitter`) to predict churn risk over time.
    4.  Save the new model as `survival_model.pkl`.
*   **Self-Correction & Refactoring:**
    *   **Observation:** I initially placed feature engineering logic directly in the training script, violating our project's principle of centralized data processing.
    *   **Action:** Proactively refactored the code. Created a `prepare_data_for_survival` function in `data_processing.py` and updated `train_survival_model.py` to use it.
    *   **User Feedback:** The user approved of this self-correction, noting it was an intelligent and valuable step. This reinforces our focus on the "Quality of Work" and "Adherence to Work" criteria.
*   **Dashboard Integration & Design Decision:**
    *   **Action:** Integrated the survival analysis workflow into the Streamlit dashboard, allowing users to run the pipeline and see time-based retention strategies.
    *   **User Query:** The user asked why we have two separate retention strategies and if they should be merged.
    *   **Decision:** We decided to keep the two strategies separate. This is a key design choice. It allows us to showcase two distinct, powerful methodologies: one based on *why* a customer churns (XGBoost + SHAP) and one based on *when* they will churn (Survival Analysis). This demonstrates a deeper level of analysis and problem-solving.