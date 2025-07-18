from xgboost import XGBClassifier # Import XGBClassifier
from sklearn.model_selection import GridSearchCV # Import GridSearchCV
from sklearn.metrics import classification_report # For evaluation metrics

def train_and_predict(X_train, y_train, X_test):
    """
    Trains XGBoostClassifier model with hyperparameter tuning and predicts on test data.

    Args:
        X_train: training features (e.g., NumPy array or sparse matrix).
        y_train: training labels (list or NumPy array).
        X_test: test features (e.g., NumPy array or sparse matrix).

    Returns:
        y_pred: predicted labels for test set.
        best_model: The best trained XGBoostClassifier model found by GridSearchCV.
    """
    print("   - Starting XGBoost training with GridSearchCV for hyperparameter tuning...")

    # Define the parameter grid to search for XGBoostClassifier
    # These are common hyperparameters to tune. You can expand or reduce this grid.
    param_grid = {
        'n_estimators': [100, 200],       # Number of boosting rounds (trees)
        'learning_rate': [0.05, 0.1, 0.2], # Step size shrinkage to prevent overfitting
        'max_depth': [3, 5, 7],           # Maximum depth of a tree
        'subsample': [0.8, 1.0],          # Subsample ratio of the training instance
        'colsample_bytree': [0.8, 1.0],   # Subsample ratio of columns when constructing each tree
        # 'gamma': [0, 0.1, 0.2],         # Minimum loss reduction required to make a further partition
        # 'reg_alpha': [0, 0.1, 0.5]      # L1 regularization term on weights
    }

       # Initialize GridSearchCV
    # cv=5 means 5-fold cross-validation
    # scoring='f1_weighted' is a good choice for imbalanced datasets, otherwise 'accuracy'
    # n_jobs=-1 uses all available CPU cores

    GBC = XGBClassifier(random_state=42, verbosity=0, eval_metric='logloss')


    grid_search = GridSearchCV(
        estimator= GBC,
        param_grid=param_grid,
        cv=5, # 5-fold cross-validation
        scoring='f1_weighted', # Or 'accuracy', 'roc_auc', etc.
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

    # Return both the predictions and the best model object
    return y_pred, best_model