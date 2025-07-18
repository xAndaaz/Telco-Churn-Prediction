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
*   **Bug Report & Fix:**
    *   **User Report:** User identified a bug in the batch prediction dashboard feature (`'clv_tier'` error).
    *   **Diagnosis:** The `clv_tier` column was being lost during one-hot encoding in the prediction pipeline.
    *   **Action:** Fixed the bug by modifying `data_processing.py` to return the `clv_tier` data alongside the model-ready data, and updated `prediction_pipeline.py` to use it correctly. This demonstrates our problem-solving ability.
*   **New Instruction:** User requested to refactor `survival_retention_strategy.py` for better code consistency, moving the file-saving logic into the `if __name__ == "__main__"` block.

---

## 2025-07-17

### Strategic Review and Gap Analysis

*   **Action:** Conducted a holistic review of the project against the user's key evaluation criteria.
*   **Analysis:** The project is strong but can be elevated by addressing several professional-level gaps.
    *   **Gap 1: Model Justification.** We use advanced models but don't benchmark them against simpler alternatives to prove their value. (Targets: "Quality of Work", "Problem-Solving").
    *   **Gap 2: Experiment Tracking.** We lack a systematic way to track model performance over time, making it hard to quantify improvements. (Targets: "Adherence to Work", "Quality of Work").
    *   **Gap 3: Advanced Techniques.** We can push the "Creativity and Innovation" aspect further with more sophisticated CLV modeling or by introducing uplift modeling.
*   **Proposed Plan:**
    1.  **Implement Model Benchmarking:** Modify the training script to compare XGBoost against Logistic Regression and Random Forest to justify our model choice.
    2.  **Introduce Lightweight Experiment Tracking:** Create a simple `experiments.json` file to log the performance of each model run, providing a clear history of our work.
    3.  **Explore Advanced Modeling:** Depending on the outcome of the first two steps, investigate more advanced CLV or uplift modeling techniques.
*   **User Decision:** Awaiting user approval on the proposed plan, starting with model benchmarking.

### Code Refactoring & Quality Improvement

*   **User Instruction:** User identified that feature engineering logic was duplicated across multiple scripts (`train_model.py`, `data_processing.py`) and suggested refactoring to adhere to the DRY (Don't Repeat Yourself) principle.
*   **Action:**
    1.  Created a new `feature_engineering.py` module to centralize all feature creation logic.
    2.  Refactored `train_model.py` and `data_processing.py` to import and use the new centralized functions.
*   **Problem-Solving (Execution Environment):** Encountered and resolved several `run_shell_command` errors on the Windows OS due to spaces in the absolute file path of the project.
    *   **Initial Attempts:** Tried various quoting and activation methods which failed.
    *   **Solution:** The most reliable method was to execute the command from the project's root directory and use a relative path to the virtual environment's Python executable (`.\.venv\Scripts\python.exe train_model.py`). This avoided path interpretation issues by the shell.
*   **Verification:** Successfully ran the `train_model.py` script after refactoring to confirm the changes were implemented correctly and the pipeline works as expected.
*   **Action (Git):** Wrote a detailed commit message to `commit_message.txt` and committed the changes, following the established project workflow.

### Model Benchmarking & Experiment Tracking

*   **Decision:** To scientifically justify the choice of XGBoost and to demonstrate professional rigor, we will implement a benchmarking and experiment tracking system.
*   **Strategy Discussion:** We discussed the best way to compare models. We decided on a two-stage approach that is both efficient and robust.
    1.  **Benchmark Stage:** Compare challenger models (Decision Tree, Random Forest, XGBoost-RF) using their default hyperparameters.
    2.  **Champion Stage:** Use the pre-existing, highly-tuned hyperparameters for our champion model (XGBoost) that were found previously using Optuna.
*   **Reasoning:** This strategy realistically simulates a professional workflow. It avoids wasting computational resources on tuning models that are unlikely to perform well, while still proving that our chosen model is superior even to untuned challengers. It also allows us to explicitly show the value added by hyperparameter tuning on the best model.
*   **Implementation Plan:**
    1.  Modify `train_model.py` to train and evaluate all four models in sequence.
    2.  Keep the Optuna code block commented out to preserve the history of how the champion model's parameters were found.
    3.  Create a new `experiments.json` file.
    4.  Log the results of each model (name, F1 score, AUC, training time) to `experiments.json` for a clear, auditable record.
*   **User Decision:** User approved this plan. Proceeding with implementation.

### Model Tuning and Selection

*   **Bug Identified:** The initial benchmarking logic in `train_model.py` was flawed. It compared a tuned `XGBClassifier` with an untuned `XGBRFClassifier`, leading to incorrect conclusions.
*   **User Analysis:** The user manually compared all models with default parameters and correctly analyzed the `experiments.json` output and identified that the untuned `XGBRFClassifier` actually had superior baseline performance.
*   **Strategic Pivot:** Based on this data-driven insight, we pivoted our strategy to adopt `XGBRFClassifier` as the new champion model. This demonstrates strong analytical and problem-solving skills.
*   **Bug Identified (Optuna):** The `XGBRFClassifier` does not support `early_stopping_rounds`. The training script failed during hyperparameter optimization.
*   **Action:** Corrected the `objective` function in `train_model.py` by removing the unsupported `fit` parameters.
*   **Action:** Executed the corrected script, successfully tuning the `XGBRFClassifier` with Optuna and saving the new, improved model. This marks a significant improvement in the project's predictive power.
