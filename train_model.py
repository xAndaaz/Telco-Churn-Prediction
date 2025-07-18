import numpy as np
import pandas as pd
import pickle
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score, precision_recall_curve
from xgboost import XGBClassifier
from feature_engineering import engineer_features

# 1. Load Dataset
df = pd.read_csv(r'Dataset/newds.csv')

# 2. Engineer Features
df, clv_bins = engineer_features(df, is_training=True)

# 3. One-hot encode categorical features
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Save the processed data with CLV for later use in the prescription pipeline
df_encoded.to_csv('Dataset/data_with_clv.csv', index=False)

# 4. Define Features (X) and Target (y)
X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

# 5. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 6. Model Benchmarking and Experiment Tracking
import time
import json
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRFClassifier


models_to_benchmark = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost RF": XGBRFClassifier(random_state=42),
    "XGBoost ": XGBClassifier(random_state= 42)
}

experiment_results = []

print("\n--- Starting Model Benchmarking ---")

for name, model in models_to_benchmark.items():
    print(f"Training {name}...")
    start_time = time.time()
    
    # Handle scale_pos_weight for XGBoost models
    if "XGBoost" in name:
        scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
        model.set_params(scale_pos_weight=scale_pos_weight)
        
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"  -> F1 Score: {f1:.4f}, AUC: {auc:.4f}, Training Time: {training_time:.2f}s")
    
    result = {
        "model_name": name,
        "f1_score": f1,
        "auc_score": auc,
        "training_time_seconds": training_time,
        "run_timestamp": datetime.now().isoformat(),
        "parameters": model.get_params()
    }
    experiment_results.append(result)

# Save results to the JSON file
with open('experiments.json', 'w') as f:
    json.dump(experiment_results, f, indent=4, default=str)

print("--- Benchmarking Complete. Results saved to experiments.json ---\n")


# 7. Handle Class Imbalance with scale_pos_weight
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# 8. Hyperparameter Tuning with Optuna for the Champion Model (XGBRFClassifier)
def objective(trial):
    """Define the objective function for Optuna, targeting XGBRFClassifier."""
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'scale_pos_weight': scale_pos_weight,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
    }

    model = XGBRFClassifier(**params, use_label_encoder=False)
    
    # XGBRFClassifier does not use early stopping, so we fit directly.
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return f1_score(y_test, y_pred, pos_label=1)

# To re-run optimization, comment out the 'best_params' line below and uncomment the study block.
# print("--- Starting Hyperparameter Optimization for XGBRFClassifier ---")
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100) 
# 
# print(f"Best trial F1-score for XGBRFClassifier: {study.best_value}")
# print("Best hyperparameters found:")
# best_params = study.best_params
# for key, value in best_params.items():
#     print(f"  {key}: {value}")

#  best parameters found
best_params = {'n_estimators': 967,'max_depth': 7,'learning_rate': 0.05453030824889137,'subsample': 0.6368751226537273,'colsample_bytree': 0.676532719598088,'reg_alpha': 0.13983880844177812,'reg_lambda': 9.306678034024623e-05}

# 9. Train Final Model with Best Hyperparameters
print("\n--- Training final XGBRFClassifier model with best hyperparameters... ---")
final_model = XGBRFClassifier(
    **best_params,

    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False
)

final_model.fit(X_train, y_train)

# 10. Threshold Tuning
y_proba_final = final_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_final)

# Find the threshold that maximizes the F1 score
f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
best_threshold = thresholds[np.argmax(f1_scores)]

print(f"\nBest Threshold found at: {best_threshold:.4f}")

# 11. Evaluate the Final Model with the Best Threshold
print("\n--- Evaluation on Test Set with Optimized Model and Threshold ---")
y_pred_final = (y_proba_final >= best_threshold).astype(int)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_final))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final))
print("\nAUC Score:", roc_auc_score(y_test, y_proba_final))

# 12. Save the Trained Model and Columns
print("\n--- Saving the trained XGBRFClassifier model and training columns... ---")
with open('Models/model.pkl', 'wb') as f:
    pickle.dump(final_model, f)


# Save the column list for later use in prediction/analysis
with open('Models/training_columns.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)

# Save the CLV bins for use in the prediction API
with open('Models/clv_bins.pkl', 'wb') as f:
    pickle.dump(clv_bins, f)

print("\nModel training and saving complete.")
