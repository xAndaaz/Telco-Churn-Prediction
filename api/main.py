
import pandas as pd
import pickle
import shap
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# --- 1. SETUP ---
# Load model and columns
with open('Models/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('Models/training_columns.pkl', 'rb') as f:
    training_columns = pickle.load(f)
with open('Models/clv_bins.pkl', 'rb') as f:
    clv_bins = pickle.load(f)

# Initialize the FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="An API to predict customer churn and identify key drivers using an XGBoost model and SHAP.",
    version="1.0.0"
)

# --- 2. PYDANTIC MODEL FOR INPUT DATA ---
# This defines the structure of the request body
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# --- 3. DATA PREPARATION LOGIC ---
def prepare_data_for_prediction(df: pd.DataFrame):
    """Prepares raw data for prediction, applying the same transformations used in training."""
    
    # Calculate CLV
    assumed_acquisition_cost = 100
    df['clv'] = (df['MonthlyCharges'] * df['tenure']) - assumed_acquisition_cost
    df['clv_tier'] = pd.cut(df['clv'], bins=clv_bins, labels=['Low', 'Medium', 'High'], include_lowest=True)
    
    # Feature Engineering
    df['tenure_monthly_interaction'] = df['tenure'] * df['MonthlyCharges']
    premium_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    df['premium_services_count'] = df[premium_services].apply(lambda x: (x == 'Yes').sum(), axis=1)
    df['tenure_monthly_ratio'] = df['tenure'] / (df['MonthlyCharges'] + 1e-6)
    
    # One-hot encode categorical features
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Align columns with the training data, filling missing columns with 0
    df_aligned = df_encoded.reindex(columns=training_columns, fill_value=0)
    
    return df_aligned

# --- 4. API ENDPOINT ---
@app.post("/predict")
def predict_churn(customer_data: CustomerData):
    """
    Predicts churn for a single customer.
    
    Accepts customer data, processes it, and returns the churn prediction,
    probability, and top 3 churn drivers.
    """
    # Convert input data to a pandas DataFrame
    df = pd.DataFrame([customer_data.dict()])
    
    # Prepare the data for the model
    df_prepared = prepare_data_for_prediction(df)
    
    # Get prediction and probability
    prediction = model.predict(df_prepared)[0]
    probability = model.predict_proba(df_prepared)[:, 1][0]
    
    # Get SHAP explanations for the top 3 drivers
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_prepared)
    
    shap_df = pd.DataFrame(shap_values, columns=df_prepared.columns)
    top_drivers = shap_df.abs().iloc[0].nlargest(3).index.tolist()
    
    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability),
        "top_churn_drivers": top_drivers
    }

# --- 5. MAIN EXECUTION (for local testing) ---
if __name__ == '__main__':
    # This allows you to run the API directly for testing
    # In production, you would use a proper ASGI server like Gunicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
