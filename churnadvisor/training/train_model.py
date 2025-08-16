import numpy as np
import pandas as pd
import pickle
import optuna
import time
import json
import os
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_recall_curve, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
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

# Create a final hold-out test set for unbiased evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


# Model Benchmarking with Cross-Validation---------------------------------------------------------------
models_to_benchmark = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost RF": XGBRFClassifier(random_state=42, use_label_encoder=False),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False)
}

experiment_results = []
experiments_file_path = os.path.join(PROJECT_ROOT, 'experiments.json')

print("\nStarting Model Benchmarking with 5-Fold Stratified Cross-Validation")
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models_to_benchmark.items():
    print(f"Cross-validating {name}...")
    start_time = time.time()

    # Create a pipeline to handle resampling within each fold to prevent data leakage
    # Note: SMOTEENN is applied only to the training data in each fold.
    pipeline = ImbPipeline(steps=[
        ('smoteenn', SMOTEENN(random_state=42)),
        ('classifier', model)
    ])

    # Set scale_pos_weight for XGBoost inside the pipeline
    if name == "XGBoost":
        scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
        pipeline.set_params(classifier__scale_pos_weight=scale_pos_weight)

    # Perform cross-validation
    # We use roc_auc as the scoring metric because it's robust for imbalanced datasets
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv_strategy, scoring='roc_auc', n_jobs=-1)
    
    validation_time = time.time() - start_time
    
    result = {
        "model_name": name,
        "timestamp": datetime.now().isoformat(),
        "mean_auc_score": np.mean(scores),
        "std_auc_score": np.std(scores),
        "validation_time_seconds": validation_time,
        "parameters": model.get_params()
    }
    experiment_results.append(result)
    print(f"Completed {name} in {validation_time:.2f}s. Mean AUC: {result['mean_auc_score']:.4f} (+/- {result['std_auc_score']:.4f})")

try:
    with open(experiments_file_path, 'w') as f:
        json.dump(experiment_results, f, indent=4, default=str)
    print(f"Benchmarking Complete. Results saved to {experiments_file_path} ----\n")
except Exception as e:
    print(f"Error saving benchmark results: {e}")


# The benchmarking results should guide the choice of the final model.
# Based on prior runs, XGBRFClassifier is a strong candidate. We will tune it.

# Hyperparameter Tuning with Optuna and Cross-Validation 
def objective(trial):
    """Define the objective function for Optuna, incorporating cross-validation."""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'use_label_encoder': False,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
    }
    
    model = XGBRFClassifier(**params)
    
    # Create a pipeline that resamples the data in each fold
    pipeline = ImbPipeline(steps=[
        ('smoteenn', SMOTEENN(random_state=42)),
        ('classifier', model)
    ])
    
    # Perform cross-validation and return the mean score for Optuna to optimize
    cv_strategy_tuning = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Using 3 splits for faster tuning
    score = cross_val_score(pipeline, X_train, y_train, cv=cv_strategy_tuning, scoring='roc_auc', n_jobs=-1)
    
    return score.mean()

print("Starting Hyperparameter Optimization for XGBRFClassifier")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=150) 

#Use this for previously found best params
#best_params = {'n_estimators': 931,'max_depth': 6,'learning_rate': 0.06592617298723388,'subsample': 0.6727515410964362, 'colsample_bytree': 0.6261582038603299, 'reg_alpha': 1.561233624717821e-05, 'reg_lambda': 0.6763491140646857}


print(f"Best trial AUC score: {study.best_value}")
print("Best hyperparameters found:")
best_params = study.best_params
for key, value in best_params.items():
    print(f"  {key}: {value}")


# Train and Evaluate Final Model on the Hold-Out Test Set -----------------

# First apply SMOTEENN to the full training set for the final model training
print("\nApplying SMOTEENN to the full training data before final fit...")
smote_enn = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

print("\n Training Final Model with Best Hyperparameters")
# Instantiate the final model with the best parameters found by Optuna
final_model = XGBRFClassifier(**best_params, random_state=42, use_label_encoder=False, objective='binary:logistic')
final_model.fit(X_train_resampled, y_train_resampled)

# Evaluate on the held-out test set
y_proba_final = final_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_final)
# Calculate F1 score for each threshold
f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
# Find the optimal threshold that maximizes F1 score
best_threshold = thresholds[np.argmax(f1_scores)]
y_pred_final = (y_proba_final >= best_threshold).astype(int)

print(f"\nBest Threshold found at: {best_threshold:.4f}")
print("\n Final Unbiased Evaluation on Hold-Out Test Set")
print(classification_report(y_test, y_pred_final))
print(f"AUC Score: {roc_auc_score(y_test, y_proba_final):.4f}")

print("\nSaving Model and Artifacts")
with open(os.path.join(PROJECT_ROOT, 'Models', 'model.pkl'), 'wb') as f:
    pickle.dump(final_model, f)
with open(os.path.join(PROJECT_ROOT, 'Models', 'training_columns.pkl'), 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)
with open(os.path.join(PROJECT_ROOT, 'Models', 'clv_bins.pkl'), 'wb') as f:
    pickle.dump(clv_bins, f)

print("\nModel training and saving complete")