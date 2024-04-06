import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

""" Section 1: Decision Trees """

# Read the CSV file into a DataFrame
df = pd.read_csv("german_credit.csv")

# Separate features (X) and target variable (y)
X = df.drop(columns=['Creditability'])
y = df['Creditability']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the decision tree classifier
clf = DecisionTreeClassifier()

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the classifier
print("Classification Report (Decision Trees):")
print(classification_report(y_test, y_pred))


""" Section 2: Boosting """
class AdaBoost(BaseEstimator, ClassifierMixin):
    """
    AdaBoost classifier implementation using decision stumps as weak learners.
    """

    def __init__(self, n_estimators=50, learning_rate=1.0):
        """
        Initialize AdaBoost classifier.

        Parameters:
        - n_estimators: Number of decision stumps to use in the ensemble.
        - learning_rate: Learning rate for the AdaBoost algorithm.
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def fit(self, X, y):
        """
        Train the AdaBoost classifier.

        Parameters:
        - X: Input features.
        - y: Target labels.
        """
        n_samples = X.shape[0]
        weights = np.ones(n_samples) / n_samples

        self.estimators_ = []
        self.alphas_ = []
        self.classes_ = np.unique(y)

        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=1)
            tree.fit(X, y, sample_weight=weights)

            predictions = tree.predict(X)
            error = np.sum(weights * (predictions != y)) / np.sum(weights)

            alpha = self.learning_rate * np.log((1 - error) / error)
            self.estimators_.append(tree)
            self.alphas_.append(alpha)

            # Update weights
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

        return self

    def predict(self, X):
        """
        Make predictions using the trained AdaBoost classifier.

        Parameters:
        - X: Input features.

        Returns:
        - Predicted class labels.
        """
        weighted_sum = np.zeros(X.shape[0])
        for alpha, estimator in zip(self.alphas_, self.estimators_):
            weighted_sum += alpha * estimator.predict(X)
        return np.sign(weighted_sum)

# Perform 10-fold cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Define parameter grid for AdaBoost classifier
param_grid_adaboost = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.1, 0.5, 1.0]
}

# Create AdaBoost instance
boosted_model = AdaBoost()

# Perform grid search for AdaBoost classifier
grid_search_adaboost = GridSearchCV(boosted_model, param_grid_adaboost, cv=cv, scoring='accuracy')
grid_search_adaboost.fit(X_train, y_train)

# Get best parameters and best score
best_params_adaboost = grid_search_adaboost.best_params_
best_score_adaboost = grid_search_adaboost.best_score_

# Print best parameters and best score
print("Best Parameters (AdaBoost):", best_params_adaboost)
print("Best Score (AdaBoost):", best_score_adaboost)

# Make predictions using the best model
best_adaboost = grid_search_adaboost.best_estimator_
y_pred_adaboost = best_adaboost.predict(X_test)

# Evaluate the classifier
print("Classification Report (AdaBoost):")
print(classification_report(y_test, y_pred_adaboost, zero_division=1))
