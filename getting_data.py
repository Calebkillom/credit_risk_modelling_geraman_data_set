import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier
import numpy as np

""" Section 1: Decision Trees """

# Read the CSV file into a DataFrame
df = pd.read_csv("german_credit.csv")

# Separate features (X) and target variable (y)
X = df.drop(columns=['Creditability'])
y = df['Creditability']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values by imputing with the mean for continuous features
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Initialize the decision tree classifier with Gini index as the splitting criterion and maximum depth
clf = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=42)

# Fit the classifier to the training data
clf.fit(X_train_imputed, y_train)

# Make predictions on the test data
y_pred_dt = clf.predict(X_test_imputed)

# Evaluate the classifier
print("Classification Report (Decision Trees):")
print(classification_report(y_test, y_pred_dt))

""" Section 2: AdaBoost with Performance Tuning """

# Initialize AdaBoost classifier
ada_boost = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), algorithm='SAMME')


# Define parameter grid for AdaBoost classifier
param_grid_adaboost = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.1, 0.5, 1.0]
}

# Perform grid search for AdaBoost classifier
grid_search_adaboost = GridSearchCV(ada_boost, param_grid_adaboost, cv=10)
grid_search_adaboost.fit(X_train_imputed, y_train)

# Get best parameters and best score
best_params_adaboost = grid_search_adaboost.best_params_
best_score_adaboost = grid_search_adaboost.best_score_

# Print best parameters and best score
print("\nBest Parameters (AdaBoost):", best_params_adaboost)
print("Best Score (AdaBoost):", best_score_adaboost)

# Make predictions using the best model
best_adaboost = grid_search_adaboost.best_estimator_
y_pred_adaboost = best_adaboost.predict(X_test_imputed)

# Evaluate the classifier
print("\nClassification Report (AdaBoost):")
print(classification_report(y_test, y_pred_adaboost))
