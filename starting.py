import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Generate a larger synthetic dataset
X, y = make_classification(
    n_samples=5000,  # Larger sample size
    n_features=20,   # More features
    n_informative=10,
    n_redundant=5,
    n_classes=3,     # Multi-class classification
    random_state=42
)

# Create a DataFrame for better handling
data = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(1, 21)])
data['Target'] = y

# Display dataset information
print("Dataset Info:\n", data.describe())

# PCA for Dimensionality Reduction (reducing features to 2 for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
pca_data = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_data['Target'] = y

# Scatter plot of the first two principal components
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_data, x='PC1', y='PC2', hue='Target', palette='viridis', alpha=0.7)
plt.title("PCA Scatter Plot of Data")
plt.show()

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest with Grid Search for Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters and performance
best_clf = grid_search.best_estimator_
print("\nBest Parameters Found:", grid_search.best_params_)

# Predictions and Evaluation
y_pred = best_clf.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(best_clf, X_test, y_test, cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Feature Importance Visualization
importances = best_clf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = data.columns[:-1]

plt.figure(figsize=(12, 6))
sns.barplot(x=importances[indices], y=feature_names[indices])
plt.title("Feature Importances from Best Random Forest Model")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()
