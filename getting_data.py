import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

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
print("Classification Report:")
print(classification_report(y_test, y_pred))

class AdaBoost:
    """
    AdaBoost classifier implementation using decision stumps as weak learners.
    """

    def __init__(self, n_estimators=50):
        """
        Initialize AdaBoost classifier.

        Parameters:
        - n_estimators: Number of decision stumps to use in the ensemble.
        """
        self.n_estimators = n_estimators
        self.estimators = []
        self.alphas = []

    def fit(self, X, y):
        """
        Train the AdaBoost classifier.

        Parameters:
        - X: Input features.
        - y: Target labels.
        """
        n_samples = X.shape[0]
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=1)
            tree.fit(X, y, sample_weight=weights)

            predictions = tree.predict(X)
            error = np.sum(weights * (predictions != y)) / np.sum(weights)

            alpha = 0.5 * np.log((1 - error) / error)
            self.estimators.append(tree)
            self.alphas.append(alpha)

            # Update weights
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

    def predict(self, X):
        """
        Make predictions using the trained AdaBoost classifier.

        Parameters:
        - X: Input features.

        Returns:
        - Predicted class labels.
        """
        weighted_sum = np.zeros(X.shape[0])
        for alpha, estimator in zip(self.alphas, self.estimators):
            weighted_sum += alpha * estimator.predict(X)
        return np.sign(weighted_sum)

boosted_model = AdaBoost(n_estimators=100)
boosted_model.fit(X_train, y_train)
predictions = boosted_model.predict(X_test)

# Evaluate the boosted model
print("Classification Report (AdaBoost):")
print(classification_report(y_test, predictions, zero_division=1))