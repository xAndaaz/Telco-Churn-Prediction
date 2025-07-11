
import pandas as pd
import pickle
import shap

# Load the trained model and training columns
with open('Models/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('Models/training_columns.pkl', 'rb') as f:
    training_columns = pickle.load(f)

def prepare_data_for_prediction(df):
    """Prepares raw data for prediction."""
    
    # Use the same CLV calculation as in training
    assumed_acquisition_cost = 100
    df['clv'] = (df['MonthlyCharges'] * df['tenure']) - assumed_acquisition_cost
    df['clv_tier'] = pd.qcut(df['clv'], q=3, labels=['Low', 'Medium', 'High'])
    
    # Recreate the same engineered features
    df['tenure_monthly_interaction'] = df['tenure'] * df['MonthlyCharges']
    premium_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    df['premium_services_count'] = df[premium_services].apply(lambda x: (x == 'Yes').sum(), axis=1)
    df['tenure_monthly_ratio'] = df['tenure'] / (df['MonthlyCharges'] + 1e-6)
    
    # One-hot encode categorical features
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Align columns with the training data
    df_aligned = df_encoded.reindex(columns=training_columns, fill_value=0)
    
    return df_aligned

def get_predictions_and_explanations(df):
    """Get predictions and SHAP explanations for the input data."""
    
    # Prepare the data
    df_prepared = prepare_data_for_prediction(df.copy())
    
    # Get predictions
    predictions = model.predict(df_prepared)
    probabilities = model.predict_proba(df_prepared)[:, 1]
    
    # Get SHAP explanations
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_prepared)
    
    # Combine results into a single DataFrame
    results = df.copy()
    results['churn_prediction'] = predictions
    results['churn_probability'] = probabilities
    
    # Get top 3 features for each prediction
    shap_df = pd.DataFrame(shap_values, columns=df_prepared.columns)
    top_features = shap_df.abs().apply(lambda x: x.nlargest(3).index.tolist(), axis=1)
    results['top_churn_drivers'] = top_features
    
    return results

if __name__ == '__main__':
    # Load sample data
    sample_df = pd.read_csv('Dataset/sample_test.csv')
    
    # Get predictions and explanations
    prediction_results = get_predictions_and_explanations(sample_df)
    
    # Calculate CLV and CLV tier for the results dataframe
    assumed_acquisition_cost = 100
    prediction_results['clv'] = (prediction_results['MonthlyCharges'] * prediction_results['tenure']) - assumed_acquisition_cost
    prediction_results['clv_tier'] = pd.qcut(prediction_results['clv'], q=3, labels=['Low', 'Medium', 'High'])

    # Save the results to a CSV file
    prediction_results.to_csv('Dataset/retention_candidates.csv', index=False)

    # Display results
    print("Churn Predictions and Top Drivers:")
    print(prediction_results[['customerID', 'clv', 'clv_tier', 'churn_prediction', 'churn_probability', 'top_churn_drivers']])
