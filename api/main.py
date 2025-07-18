import pandas as pd
import pickle
import shap
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import sys
import os

# Add the parent directory to the path to allow importing from the root folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_processing import prepare_data_for_prediction

# --- 1. SETUP ---
# Load model and explainer at startup
model_path = os.path.join(os.path.dirname(__file__), '..', 'Models', 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

explainer = shap.TreeExplainer(model)

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

# --- 3. API ENDPOINT ---
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
    # Note: The function now returns two dataframes, we only need the first for the model
    df_prepared, _ = prepare_data_for_prediction(df)
    
    # Get prediction and probability
    prediction = model.predict(df_prepared)[0]
    probability = model.predict_proba(df_prepared)[:, 1][0]
    
    # Get SHAP explanations using the pre-loaded explainer
    shap_values = explainer.shap_values(df_prepared)
    
    shap_df = pd.DataFrame(shap_values, columns=df_prepared.columns)
    top_drivers = shap_df.abs().iloc[0].nlargest(3).index.tolist()
    
    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability),
        "top_churn_drivers": top_drivers
    }

# 5. MAIN EXECUTION (for local testing)
if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
