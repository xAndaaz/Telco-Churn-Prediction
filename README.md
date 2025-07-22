# ChurnAdvisor: Customer Churn Prediction & Risk Analysis

![MIT License](https://img.shields.io/badge/License-MIT-green.svg)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://telco-churn-advisor-8.streamlit.app)

**ChurnAdvisor** is an end-to-end machine learning application that identifies at-risk customers and generates a holistic **Churn Risk Profile** for each one. Instead of prescribing generic actions, this tool provides a rich, multi-faceted advisory report that empowers business users to make informed, data-driven retention decisions.

---

## 🚀 Live Demo & Preview

**[➡️ View the Live Application on Streamlit Community Cloud](https://telco-churn-advisor-8.streamlit.app)**

![ChurnAdvisor GIF Demo](https://github.com/user-attachments/assets/YOUR_GIF_ASSET_URL_HERE)
*(**Recommendation:** Create a short, high-quality GIF showcasing the app's workflow—uploading a file, viewing the results, and using the Instant Prediction—and replace the link above.)*

---

## ✨ Key Features

*   **Professional Project Structure:** All code is organized into a standard, maintainable Python package (`churnadvisor`).
*   **Dual-Model System:** Combines a high-precision **XGBRFClassifier** (to predict *if* and *why*) with a **Cox Proportional Hazards** survival model (to predict *when*).
*   **Context-Aware Insight Engine:** A sophisticated analysis module that intelligently interprets model outputs (SHAP drivers) in the context of each customer's actual data to generate factually consistent, actionable insights.
*   **Robust Interactive Dashboard:** A user-friendly Streamlit application (`ChurnAdvisor`) that runs as a single, self-contained process. It features:
    *   **Instant Prediction:** Real-time analysis for a single customer.
    *   **Batch Analysis:** Bulk processing of customer lists from a CSV file.
    *   **Advanced Visualization:** An interactive SHAP beeswarm plot to visualize global feature impact.

---

## 🛠️ Tech Stack

*   **Language:** Python
*   **Application Framework:** Streamlit
*   **Core ML Libraries:** Scikit-learn, XGBoost, Lifelines (for Survival Analysis), Imbalanced-learn(SMOTEEN)
*   **Explainable AI (XAI):** SHAP
*   **Data Manipulation:** Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn

---

## ⚙️ How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/xAndaaz/telco-churn-prediction.git
cd telco-churn-prediction
```

### 2. Set Up the Environment
It is recommended to use a virtual environment.
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Train the Models
Execute the training scripts from the project's root directory:
```bash
# Train the classification model
python -m churnadvisor.training.train_model

# Train the survival model
python -m churnadvisor.training.train_survival_model
```

### 4. Launch the Application
*   **Start the dashboard from the project root directory:**
    ```bash
    streamlit run dashboard/app.py
    ```

---

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
*For a detailed technical guide, including the project structure and data flow, please see `devReadme.md`.*
