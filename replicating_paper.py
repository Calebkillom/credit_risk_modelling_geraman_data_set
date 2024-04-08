import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import wilcoxon
from scipy.stats import chi2

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

""" Section 5: Plotting the ROC """
# Define models and their predictions
models = {
    'Decision Trees with Pruning': pruned_tree,
    'AdaBoost': best_adaboost,
    'MLP': best_mlp,
    'SVM': best_svm
}

# Plot ROC curves for each model
plt.figure(figsize=(8, 8))
for model_name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test_imputed)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        # Interpolate the ROC curve for smoother visualization
        fpr_interp = np.linspace(0, 1, 1000)
        tpr_interp = np.interp(fpr_interp, fpr, tpr)
        plt.plot(fpr_interp, tpr_interp, label=f'{model_name} (AUC = {roc_auc:.2f})')

# Plot random guess line
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')

# Set plot labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)

# Show plot
plt.show()

""" Section 6: AUC Estimates """
# Define AUC estimates for each model based on reported best scores
AUC_DT = 0.69  # Estimated value based on reported accuracy
AUC_AdaBoost = 0.7675
AUC_MLP = 0.7025
AUC_SVM = 0.71875

# Define the number of samples for good and bad credit sets
ng = 150  # Number of samples for class 1
nb = 150   # Number of samples for class 0

# Calculate the Wilcoxon-Mann-Whitney statistic for each pairwise comparison
theta_MLP_SVM = AUC_MLP - AUC_SVM
theta_MLP_AdaBoost = AUC_MLP - AUC_AdaBoost
theta_SVM_AdaBoost = AUC_SVM - AUC_AdaBoost

# Calculate the variance of AUC estimators for each model
v_MLP = 1 / nb * np.sum(np.ones(nb) * ng - np.arange(1, ng + 1))
v_SVM = 1 / ng * np.sum(np.ones(ng) * nb - np.arange(1, nb + 1))
v_AdaBoost = 1 / nb * np.sum(np.ones(nb) * ng - np.arange(1, ng + 1))

# Calculate the covariance of AUC estimators for each pairwise comparison
cov_MLP_SVM = 1 / (ng * (ng - 1)) * np.sum((ng - np.arange(1, ng + 1)) * (nb - np.arange(1, nb + 1)))
cov_MLP_AdaBoost = cov_SVM_AdaBoost = 0  # Since these models were not directly compared in the provided results

# Compute the test statistic (T) for each pairwise comparison
var_theta_MLP_SVM = v_MLP + v_SVM - 2 * cov_MLP_SVM
T_MLP_SVM = (theta_MLP_SVM ** 2) / var_theta_MLP_SVM

var_theta_MLP_AdaBoost = v_MLP + v_AdaBoost - 2 * cov_MLP_AdaBoost
T_MLP_AdaBoost = (theta_MLP_AdaBoost ** 2) / var_theta_MLP_AdaBoost

var_theta_SVM_AdaBoost = v_SVM + v_AdaBoost - 2 * cov_SVM_AdaBoost
T_SVM_AdaBoost = (theta_SVM_AdaBoost ** 2) / var_theta_SVM_AdaBoost

# Calculate the p-values for each pairwise comparison
p_value_MLP_SVM = 1 - chi2.cdf(T_MLP_SVM, 1)  # Using Chi-square distribution with 1 degree of freedom
p_value_MLP_AdaBoost = 1 - chi2.cdf(T_MLP_AdaBoost, 1)
p_value_SVM_AdaBoost = 1 - chi2.cdf(T_SVM_AdaBoost, 1)

# Print the results
# Print the results
print("MLP vs SVM: T-statistic =", T_MLP_SVM, "p-value =", p_value_MLP_SVM)
print("MLP vs AdaBoost: T-statistic =", T_MLP_AdaBoost, "p-value =", p_value_MLP_AdaBoost)
print("SVM vs AdaBoost: T-statistic =", T_SVM_AdaBoost, "p-value =", p_value_SVM_AdaBoost)