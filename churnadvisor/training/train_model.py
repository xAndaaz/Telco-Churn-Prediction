import numpy as np
import pandas as pd
import pickle
import optuna
import time
import json
import os
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_recall_curve, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from imblearn.combine import SMOTEENN
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from churnadvisor.processing.feature_engineering import engineer_features
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Load and Prepare Data
print("Loading and Preparing Data")
df = pd.read_csv(os.path.join(PROJECT_ROOT, 'Dataset', 'newds.csv'))
df, clv_bins = engineer_features(df, is_training=True)

categorical_cols = df.select_dtypes(include=['object', 'category']).columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

df_encoded.to_csv(os.path.join(PROJECT_ROOT, 'Dataset', 'data_with_clv.csv'), index=False)

X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


# Model Benchmarking---------------------------------------------------------------
models_to_benchmark = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost RF": XGBRFClassifier(random_state=42, use_label_encoder=False),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False)
}

experiment_results = []
experiments_file_path = os.path.join(PROJECT_ROOT, 'experiments.json')

print("\nStarting Model Benchmarking (on original data)")
for name, model in models_to_benchmark.items():
    print(f"Training {name}...")
    start_time = time.time()
    # Set scale_pos_weight only for the standard XGBoost models
    if name == "XGBoost":
        scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
        model.set_params(scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] 

    result = {
        "model_name": name,
        "timestamp": datetime.now().isoformat(),
        "f1_score": f1_score(y_test, y_pred),
        "auc_score": roc_auc_score(y_test, y_proba),
        "training_time_seconds": training_time,
        "parameters": model.get_params()
    }
    experiment_results.append(result)
    print(f"Completed {name} in {training_time:.2f}s. F1: {result['f1_score']:.4f}, AUC: {result['auc_score']:.4f}")

try:
    with open(experiments_file_path, 'w') as f:
        json.dump(experiment_results, f, indent=4, default=str)
    print(f"--- Benchmarking Complete. Results saved to {experiments_file_path} ---\n")
except Exception as e:
    print(f"Error saving benchmark results: {e}")


# Apply SMOTEENN
print("\nApplying SMOTEENN to balance the training data------")
smote_enn = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

# Hyperparameter Tuning with Optuna
# def objective(trial):
#     '''Define the objective function for Optuna, targeting XGBRFClassifier.'''
#     params = {
#         'objective': 'binary:logistic',
#         'eval_metric': 'logloss',
#         'random_state': 42,
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
#         'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
#     }
#     model = XGBRFClassifier(**params, use_label_encoder=False)
#     model.fit(X_train_resampled, y_train_resampled) # Fit on resampled data
#     y_pred = model.predict(X_test)
#     return f1_score(y_test, y_pred, pos_label=1)

# print("--- Starting Hyperparameter Optimization for XGBRFClassifier ---")
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100) # n_trials can be increased for a more thorough search

# print(f"Best trial F1-score for XGBRFClassifier: {study.best_value}")
# print("Best hyperparameters found:")
# best_params = study.best_params
# for key, value in best_params.items():
#     print(f"  {key}: {value}")
best_params = {'n_estimators': 973, 'max_depth': 7, 'learning_rate': 0.2830398857050143, 'subsample': 0.6683129550503917, 'colsample_bytree': 0.8107058913946676, 'reg_alpha': 4.1790179159967865e-05, 'reg_lambda': 0.000140473293166153}


# Train and Evaluate Final Model 
print("\n--- Training and Evaluating Final Model ---")
final_model = XGBRFClassifier(**best_params, random_state=42, use_label_encoder=False)
final_model.fit(X_train_resampled, y_train_resampled)

y_proba_final = final_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_final)
f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
best_threshold = thresholds[np.argmax(f1_scores)]
y_pred_final = (y_proba_final >= best_threshold).astype(int)

print(f"\nBest Threshold found at: {best_threshold:.4f}")
print("\n--- Final Evaluation on Test Set ---")
print(classification_report(y_test, y_pred_final))
print(f"AUC Score: {roc_auc_score(y_test, y_proba_final):.4f}")

print("\n--- Saving Model and Artifacts ---")
with open(os.path.join(PROJECT_ROOT, 'Models', 'model.pkl'), 'wb') as f:
    pickle.dump(final_model, f)
with open(os.path.join(PROJECT_ROOT, 'Models', 'training_columns.pkl'), 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)
with open(os.path.join(PROJECT_ROOT, 'Models', 'clv_bins.pkl'), 'wb') as f:
    pickle.dump(clv_bins, f)

print("\nModel training and saving complete.")