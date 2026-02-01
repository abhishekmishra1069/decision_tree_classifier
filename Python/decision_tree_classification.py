# Decision Tree Classification

# Importing the necessary libraries for data manipulation, visualization, and machine learning
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset from a CSV file
# The dataset contains features like Age and Estimated Salary, and the target variable (Purchased)
dataset = pd.read_csv('Social_Network_Ads.csv')
# Extracting features (Age, Estimated Salary) and target (Purchased)
X = dataset.iloc[:, :-1].values  # Features: all columns except the last
y = dataset.iloc[:, -1].values   # Target: last column

# Splitting the dataset into the Training set and Test set
# Using 75% for training and 25% for testing, with a fixed random state for reproducibility
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
print(X_train)  # Print training features
print(y_train)  # Print training labels
print(X_test)   # Print test features
print(y_test)   # Print test labels

# Feature Scaling
# Standardizing the features to have mean 0 and variance 1 for better model performance
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # Fit and transform training data
X_test = sc.transform(X_test)        # Transform test data using the same scaler
print(X_train)  # Print scaled training features
print(X_test)   # Print scaled test features

# Training the Decision Tree Classification model on the Training set
# Using entropy as the criterion for splitting, and random_state for reproducibility
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)  # Train the model

# Save the trained model and scaler for later use in the Flask app
import joblib
joblib.dump(classifier, 'decision_tree_model.pkl')
joblib.dump(sc, 'scaler.pkl')

# Predicting a new result
# Example prediction for a user with Age=30 and Estimated Salary=87000
# Note: Input must be scaled using the same scaler
print(classifier.predict(sc.transform([[30, 87000]])))

# Predicting the Test set results
# Generate predictions for the test set
y_pred = classifier.predict(X_test)
# Concatenate predictions and actual values for comparison
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
# Evaluate the model using confusion matrix and accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)  # Confusion matrix
print(cm)  # Print confusion matrix
print(accuracy_score(y_test, y_pred))  # Print accuracy score

# Visualising the Training set results
# Create a visualization of the decision boundary on the training set
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train  # Inverse transform to original scale for plotting
# Create a meshgrid for plotting the decision boundary
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
                     np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
# Predict on the meshgrid points
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
# Scatter plot of training data points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Decision Tree Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
# Similar visualization for the test set
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
                     np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Decision Tree Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()