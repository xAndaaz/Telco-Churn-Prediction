import numpy as np
import pandas as pd
import pickle
import optuna
import time
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score, precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from feature_engineering import engineer_features
from imblearn.combine import SMOTEENN

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
models_to_benchmark = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost RF": XGBRFClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
}

experiment_results = []
print("\n--- Starting Model Benchmarking (on original data) ---")
for name, model in models_to_benchmark.items():
    print(f"Training {name}...")
    start_time = time.time()
    
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

with open('experiments.json', 'w') as f:
    json.dump(experiment_results, f, indent=4, default=str)
print("--- Benchmarking Complete. Results saved to experiments.json ---\n")

# 7. Apply SMOTEENN to the training data
print("\n--- Applying SMOTEENN to balance the training data... ---")
smote_enn = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)
print("Original training set shape: %s" % str(X_train.shape))
print("Resampled training set shape: %s" % str(X_train_resampled.shape))

# 8. Hyperparameter Tuning with Optuna for the Champion Model (XGBRFClassifier)
def objective(trial):
    """Define the objective function for Optuna, targeting XGBRFClassifier."""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
    }
    model = XGBRFClassifier(**params, use_label_encoder=False)
    model.fit(X_train_resampled, y_train_resampled) # Fit on resampled data
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred, pos_label=1)

#To re-run optimization, comment out the 'best_params' line below and uncomment the study block.
# print("--- Starting Hyperparameter Optimization for XGBRFClassifier on Resampled Data ---")
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=200) 
# print(f"Best trial F1-score for XGBRFClassifier: {study.best_value}")
# print("Best hyperparameters found:")
# best_params = study.best_params
# for key, value in best_params.items():
#     print(f"  {key}: {value}")

# previous best params
#best_params = {'n_estimators': 642, 'max_depth': 7, 'learning_rate': 0.18834241411982827, 'subsample': 0.7047092188797366, 'colsample_bytree': 0.6429863018764423, 'reg_alpha': 0.0002094128096819643, 'reg_lambda': 2.2981513633678433e-06}
## p = 0.6, r = 0.67, 0.8318

#best_params = {'n_estimators': 822, 'max_depth': 6, 'learning_rate': 0.2536371549453208, 'subsample': 0.7909034480434396, 'colsample_bytree': 0.963443169233868, 'reg_alpha': 8.488962288691769e-07, 'reg_lambda': 2.560114435198286e-06}
# p = 0.58, r = 0.69, auc = 0.83

#best_params = {'n_estimators': 820, 'max_depth': 10, 'learning_rate': 0.27890954332537937, 'subsample': 0.7815315241923265, 'colsample_bytree': 0.9381450466757373, 'reg_alpha': 0.15708773214852248, 'reg_lambda': 0.9428192561354662}
# p = 0.55, r = 0.71, auc = 0.826

best_params = {'n_estimators': 973, 'max_depth': 7, 'learning_rate': 0.2830398857050143, 'subsample': 0.6683129550503917, 'colsample_bytree': 0.8107058913946676, 'reg_alpha': 4.1790179159967865e-05, 'reg_lambda': 0.000140473293166153}
# p = 0.6, r = 0.67, auc = 0.8317

# 9. Train Final Model with Best Hyperparameters on Resampled Data
print("\n--- Training final XGBRFClassifier model with best hyperparameters on resampled data... ---")
final_model = XGBRFClassifier(
    **best_params,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    use_label_encoder=False
    # Note: scale_pos_weight is removed as SMOTEENN handles the imbalance
)
final_model.fit(X_train_resampled, y_train_resampled)

# 10. Threshold Tuning
y_proba_final = final_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_final)

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

with open('Models/training_columns.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)

with open('Models/clv_bins.pkl', 'wb') as f:
    pickle.dump(clv_bins, f)

print("\nModel training and saving complete.")