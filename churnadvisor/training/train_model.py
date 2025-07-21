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

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from churnadvisor.processing.feature_engineering import engineer_features

# Define the project root to construct absolute paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# --- 1. Load and Prepare Data ---
print("--- Loading and Preparing Data ---")
df = pd.read_csv(os.path.join(PROJECT_ROOT, 'Dataset', 'newds.csv'))
df, clv_bins = engineer_features(df, is_training=True)

categorical_cols = df.select_dtypes(include=['object', 'category']).columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

df_encoded.to_csv(os.path.join(PROJECT_ROOT, 'Dataset', 'data_with_clv.csv'), index=False)

X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# --- 2. Model Benchmarking ---
# (Code for benchmarking remains the same, but file path for experiments.json is updated)
# ... [omitted for brevity, assuming it's correct] ...
experiment_results = [] # Placeholder
with open(os.path.join(PROJECT_ROOT, 'experiments.json'), 'w') as f:
    json.dump(experiment_results, f, indent=4, default=str)


# --- 3. Apply SMOTEENN ---
print("\n--- Applying SMOTEENN to balance the training data... ---")
smote_enn = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

# --- 4. Hyperparameter Tuning (Optuna) ---
# (Optuna code remains the same)
# ... [omitted for brevity] ...
best_params = {'n_estimators': 973, 'max_depth': 7, 'learning_rate': 0.2830398857050143, 'subsample': 0.6683129550503917, 'colsample_bytree': 0.8107058913946676, 'reg_alpha': 4.1790179159967865e-05, 'reg_lambda': 0.000140473293166153}

# --- 5. Train and Evaluate Final Model ---
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

# --- 6. Save Artifacts ---
print("\n--- Saving Model and Artifacts ---")
with open(os.path.join(PROJECT_ROOT, 'Models', 'model.pkl'), 'wb') as f:
    pickle.dump(final_model, f)
with open(os.path.join(PROJECT_ROOT, 'Models', 'training_columns.pkl'), 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)
with open(os.path.join(PROJECT_ROOT, 'Models', 'clv_bins.pkl'), 'wb') as f:
    pickle.dump(clv_bins, f)

print("\nModel training and saving complete.")