from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def vectorize(texts):
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

    return X_tf