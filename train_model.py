import numpy as np
import pandas as pd
import pickle
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score, precision_recall_curve
from xgboost import XGBClassifier

# 1. Load Dataset
df = pd.read_csv(r'Dataset/newds.csv')

# 2. Calculate Customer Lifetime Value (CLV)
# For this portfolio project, we'll use a simplified CLV formula.
# CLV = (customer['MonthlyCharges'] * customer['tenure']) - (assumed_acquisition_cost)
assumed_acquisition_cost = 100
df['clv'] = (df['MonthlyCharges'] * df['tenure']) - assumed_acquisition_cost

# Segment customers into value tiers based on CLV
_, clv_bins = pd.qcut(df['clv'], q=3, labels=['Low', 'Medium', 'High'], retbins=True, duplicates='drop')
df['clv_tier'] = pd.cut(df['clv'], bins=clv_bins, labels=['Low', 'Medium', 'High'], include_lowest=True)

# 3. Advanced Feature Engineering
df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})

# Interaction features
df['tenure_monthly_interaction'] = df['tenure'] * df['MonthlyCharges']

# Service utilization
premium_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
df['premium_services_count'] = df[premium_services].apply(lambda x: (x == 'Yes').sum(), axis=1)

# Tenure to monthly charges ratio
df['tenure_monthly_ratio'] = df['tenure'] / (df['MonthlyCharges'] + 1e-6)

# Tenure per premium service
df['tenure_per_premium_service'] = df['tenure'] / (df['premium_services_count'] + 1e-6)

# One-hot encode categorical features, including the new clv_tier
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Save the processed data with CLV for later use in the prescription pipeline
df_encoded.to_csv('Dataset/data_with_clv.csv', index=False)

# 4. Define Features (X) and Target (y)
X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

# 5. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 6. Handle Class Imbalance with scale_pos_weight
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# 6. Hyperparameter Tuning with Optuna and Cross-Validation
def objective(trial):
    """Define the objective function for Optuna."""
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'scale_pos_weight': scale_pos_weight,
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
    }

    model = XGBClassifier(**params, use_label_encoder=False)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        f1_scores.append(f1_score(y_val_fold, y_pred, pos_label=1))
    
    return np.mean(f1_scores)

# print("Starting hyperparameter optimization with Optuna (100 trials, 5-fold CV)...")
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=300)

# print(f"Best trial F1-score (CV): {study.best_value}")
# print("Best hyperparameters found:")
# for key, value in study.best_params.items():
#     print(f"  {key}: {value}")
best_params= {'n_estimators': 1015, 'max_depth': 8, 'learning_rate': 0.010749966721711619, 'subsample': 0.7538033573276144, 'colsample_bytree': 0.754614993636156, 'gamma': 9.192659488189232, 'min_child_weight': 2}

# 7. Train Final Model with Best Hyperparameters
print("\nTraining final model with best hyperparameters...")
#best_params = study.best_params
final_model = XGBClassifier(
    **best_params,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False
)

final_model.fit(X_train, y_train)

# 8. Threshold Tuning
y_proba_final = final_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_final)

# Find the threshold that maximizes the F1 score
f1_scores = (2 * precision * recall) / (precision + recall)
best_threshold = thresholds[np.argmax(f1_scores)]

print(f"\nBest Threshold: {best_threshold}")

# 9. Evaluate the Final Model with the Best Threshold
print("\nEvaluation on Test Set with Optimized Model and Threshold:")
y_pred_final = (y_proba_final >= best_threshold).astype(int)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_final))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final))
print("\nAUC Score:", roc_auc_score(y_test, y_proba_final))

# 10. Save the Trained Model and Columns
print("\nSaving the trained XGBoost model and training columns...")
with open('Models/model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

# Save the column list for later use in prediction/analysis
with open('Models/training_columns.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)

# Save the CLV bins for use in the prediction API
with open('Models/clv_bins.pkl', 'wb') as f:
    pickle.dump(clv_bins, f)

print("\nModel training and saving complete.")
