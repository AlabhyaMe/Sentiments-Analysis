from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def vectorize_train(texts):
    """
    Generates pure Term Frequency (TF) features for the entire dataset.

    Args:
        texts (list[str]): List of all preprocessed documents.

    Returns:
        np.ndarray: TF feature matrix (dense).
    """
    print("   - Generating TF features...")

    # Step 1 - Fit BoW model
    vectorizer = CountVectorizer()
    X_counts = vectorizer.fit_transform(texts)    # sparse matrix

    # Step 2 - Convert to dense matrix
    X_counts_dense = X_counts.toarray()

    # Step 3 - Normalize by row sum to get TF
    row_sums = X_counts_dense.sum(axis=1, keepdims=True)

    # Avoid division by zero
    row_sums[row_sums == 0] = 1

    X_tf = X_counts_dense / row_sums
    
    # The 'vectorizer' object (CountVectorizer instance) is returned here
    # because it is 'fitted' on the training data and contains the learned
    # vocabulary. This fitted object is essential to consistently transform
    # new, unseen data into the same feature space.
    return X_tf,vectorizer


def vectorize_test(texts, fitted_vectorizer):
    """
    Transform test data using fitted Bag-of-Words vectorizer
    and generate normalized Term Frequency (TF) features.
    """
    print("   - Transforming test data using fitted Bag-of-Words vectorizer...")
    X_counts = fitted_vectorizer.transform(texts)
    X_counts_dense = X_counts.toarray()

    row_sums = X_counts_dense.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero

    X_tf = X_counts_dense / row_sums
    return X_tf