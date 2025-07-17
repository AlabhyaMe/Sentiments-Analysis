# MLAlgo/logistic_regression_model.py

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV # Import GridSearchCV
from sklearn.metrics import classification_report # For evaluation metrics
import numpy as np # For type hinting

def train_and_predict(X_train, y_train, X_test):
    """
    Trains Logistic Regression model with hyperparameter tuning and predicts on test data.

    Args:
        X_train: training features (e.g., NumPy array or sparse matrix).
        y_train: training labels (list or NumPy array).
        X_test: test features (e.g., NumPy array or sparse matrix).

    Returns:
        y_pred: predicted labels for test set.
        best_model: The best trained LogisticRegression model found by GridSearchCV.
    """
    print("   - Starting Logistic Regression training with GridSearchCV for hyperparameter tuning...")

    # Define the parameter grid to search
    param_grid = {
        'solver': ['liblinear', 'lbfgs'], # Different solvers
        'C': [0.1, 1.0, 10.0],            # Inverse of regularization strength
        'class_weight': [None, 'balanced'], # Handle class imbalance
        'max_iter': [500, 1000]           # Number of iterations
    }

    # Initialize the Logistic Regression model
    lr_model = LogisticRegression(random_state=42)

    # Initialize GridSearchCV
    # cv=5 means 5-fold cross-validation
    # scoring='f1_weighted' is a good choice for imbalanced datasets, otherwise 'accuracy'
    # n_jobs=-1 uses all available CPU cores
    grid_search = GridSearchCV(
        estimator=lr_model,
        param_grid=param_grid,
        cv=5, # 5-fold cross-validation
        scoring='f1_weighted', # Or 'accuracy', 'roc_auc', etc., depending on your evaluation metric
        n_jobs=-1, # Use all available cores
        verbose=1 # Print progress messages
    )

    # Fit GridSearchCV to the training data
    # This will perform the hyperparameter search
    grid_search.fit(X_train, y_train)

    # Get the best model found by GridSearchCV
    best_model = grid_search.best_estimator_

    print("\n   - Best Hyperparameters found:")
    print(grid_search.best_params_)
    print(f"   - Best Cross-Validation Score (F1-weighted): {grid_search.best_score_:.4f}")

    # Make predictions on the test set using the best model
    y_pred = best_model.predict(X_test)
    print("Best model parameters:", best_model.get_params())
    
    return y_pred,best_model