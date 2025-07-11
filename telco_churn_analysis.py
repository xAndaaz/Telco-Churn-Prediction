import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

"""
Exploratory Data Analysis (EDA) for Telco Customer Churn dataset.
Performs data cleaning, feature exploration, churn analysis, and feature importance.
Saves cleaned dataset to 'Dataset/newds.csv'.
"""

# Plotting functions
def plot_histogram(df, column_name):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column_name], kde=True, bins=30)
    plt.title(f'Histogram of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.show()

def plot_boxplot(df, column_name):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column_name])
    plt.title(f'Boxplot of {column_name}')
    plt.xlabel(column_name)
    plt.show()

def plot_countplot(df, column_name, hue=None):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=df[column_name])
    plt.title(f'Count Plot of {column_name}' + (f' vs {hue}' if hue else ''))
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, columns):
    plt.figure(figsize=(12, 10))
    corr = df[columns].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.show()

# Load dataset
df = pd.read_csv(r'Dataset/Telco-Customer-Churn.csv')
df = df.drop(columns=['customerID'])

# Data cleaning
for col in df.select_dtypes(include='object').columns:
    print(f"{col}: {(df[col].str.strip() == '').sum()} empty string(s)")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
#print("Missing values before dropping:\n", df.isna().sum())
df.dropna(inplace=True)
#print("Missing values after dropping:\n", df.isna().sum())
df.to_csv('Dataset/newds.csv', index=False)

# Initial overview
print("Dataset Shape:", df.shape)
print("Data Types:\n", df.dtypes)
print("Summary Statistics:\n", df.describe())
print("First Few Rows:\n", df.head())

# Churn distribution
plot_countplot(df, 'Churn')

# Categorical feature exploration
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    plot_countplot(df, col, hue='Churn')

# Numeric feature distributions
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
for col in num_cols:
    plot_histogram(df, col)
    plot_boxplot(df, col)

# Correlation heatmap
plot_correlation_matrix(df, num_cols)

# Churn rate by Contract and Tenure
contract_churn = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()
print("Churn Rate by Contract:\n", contract_churn)
tenure_churn = df.groupby('tenure')['Churn'].value_counts(normalize=True).unstack()
tenure_churn.plot(title='Churn Rate by Tenure', figsize=(10, 5))
plt.ylabel("Churn Rate")
plt.show()

# Feature importance with Random Forest
df_encoded = df.copy()
for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]
model = RandomForestClassifier(random_state=42)
model.fit(X, y)
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', figsize=(10, 6), title='Feature Importances')
plt.show()