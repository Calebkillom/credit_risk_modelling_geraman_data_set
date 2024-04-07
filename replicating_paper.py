import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np

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

""" Section 1: Decision Trees with Pruning"""

# Initialize the decision tree classifier with Gini index as the splitting criterion and maximum depth
clf = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=42)

# Fit the classifier to the training data
clf.fit(X_train_imputed, y_train)

# Prune the decision tree to prevent overfitting
# You may need to adjust the pruning hyperparameters based on your data and model
pruned_tree = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
pruned_tree.fit(X_train_imputed, y_train)

# Make predictions on the test data using pruned tree
y_pred_dt_pruned = pruned_tree.predict(X_test_imputed)

# Evaluate the pruned classifier
print("Classification Report (Pruned Decision Tree):")
print(classification_report(y_test, y_pred_dt_pruned))

""" Section 2: AdaBoost with Weighted Averaging"""

# Initialize AdaBoost classifier
ada_boost = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), algorithm='SAMME')

# Define parameter grid for AdaBoost classifier
param_grid_adaboost = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.1, 0.5, 1.0, 1.5],
}

# Perform grid search with 10-fold cross-validation for AdaBoost classifier
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
grid_search_adaboost = GridSearchCV(ada_boost, param_grid_adaboost, cv=cv)
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

""" Section 3: Implement Multilayer Perceptron (MLP) """

# Initialize MLP classifier
mlp = MLPClassifier(random_state=42, activation='logistic', solver='sgd', learning_rate='constant')

# Define parameter grid for MLP classifier
param_grid_mlp = {
    'hidden_layer_sizes': [(21,)],
    'max_iter': [300, 400],
    'learning_rate_init': [0.0001, 0.0005, 0.001, 0.005],
}

# Perform grid search with 10-fold cross-validation for MLP classifier
grid_search_mlp = GridSearchCV(mlp, param_grid_mlp, cv=10)
grid_search_mlp.fit(X_train_imputed, y_train)

# Get best parameters and best score
best_params_mlp = grid_search_mlp.best_params_
best_score_mlp = grid_search_mlp.best_score_

# Print best parameters and best score
print("\nBest Parameters (MLP):", best_params_mlp)
print("Best Score (MLP):", best_score_mlp)

# Make predictions using the best model
best_mlp = grid_search_mlp.best_estimator_
y_pred_mlp = best_mlp.predict(X_test_imputed)

# Evaluate the classifier using 10-fold cross-validation
cv_scores = cross_val_score(best_mlp, X_train_imputed, y_train, cv=10)
print("\nCross-Validation Scores (MLP):", cv_scores)

# Evaluate the classifier
print("\nClassification Report (MLP):")
print(classification_report(y_test, y_pred_mlp, zero_division=0))



""" Section 4: Implement Support Vector Machine (SVM) """
# Initialize SVM classifier
svm = SVC(random_state=42)

# Define exponentially growing sequences for σ and C
sigma_values = np.logspace(4, 3, 8)  # Example sequence for σ
C_values = np.logspace(-3, 4, 8)       # Example sequence for C

# Define parameter grid for SVM classifier
param_grid_svm = {
    'C': C_values,
    'gamma': 1 / (2 * sigma_values ** 2),  # Convert σ to γ for RBF kernel
    'kernel': ['rbf'],
}

# Perform grid search with 10-fold cross-validation for SVM classifier
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Perform grid search with 10-fold cross-validation for SVM classifier
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=cv)
grid_search_svm.fit(X_train_imputed, y_train)

# Get best parameters and best score
best_params_svm = grid_search_svm.best_params_
best_score_svm = grid_search_svm.best_score_

# Print best parameters and best score
print("\nBest Parameters (SVM):", best_params_svm)
print("Best Score (SVM):", best_score_svm)

# Fit the best model obtained from grid search on the entire training data
best_svm = grid_search_svm.best_estimator_
y_pred_svm = best_svm.predict(X_test_imputed)

# Print classification report
print("\nClassification Report (SVM):")
print(classification_report(y_test, y_pred_svm, zero_division=0))
