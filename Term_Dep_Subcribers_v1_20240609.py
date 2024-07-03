# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:41:27 2024

@author: riyaz
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
train_data = pd.read_csv('train.csv', delimiter=';')
test_data = pd.read_csv('test.csv', delimiter=';')

# Identify features and target
X = train_data.drop(columns='y')
y = train_data['y']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for numerical features (impute missing values, scale)
numeric_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical features (impute missing values, encode)
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Function to train and evaluate models
def train_and_evaluate(model, X_train, y_train, X_val, y_val):
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    print(f"Model: {model.__class__.__name__}")
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("Classification Report:\n", classification_report(y_val, y_pred))
    return clf

# Train and evaluate each model
trained_models = {}
for model_name, model in models.items():
    clf = train_and_evaluate(model, X_train, y_train, X_val, y_val)
    trained_models[model_name] = clf

# Choose the best model based on evaluation
# Here we assume Gradient Boosting performed best
best_model = trained_models['Gradient Boosting']

# Apply preprocessing to the test data and make predictions
X_test = test_data.drop(columns='y')
y_test = test_data['y']
X_test_processed = preprocessor.transform(X_test)
y_test_pred = best_model.predict(X_test_processed)

# Output predictions
print("Test Set Predictions:")
print(y_test_pred)
print("Test Set Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test Set Classification Report:\n", classification_report(y_test, y_test_pred))
