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
