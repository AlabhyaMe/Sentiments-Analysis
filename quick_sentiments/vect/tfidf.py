# Vect/tfidf_vectorizer.py

from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize(texts: list[str]):
    """
    Generates TF-IDF features for the entire dataset.

    Args:
        texts (list[str]): List of all preprocessed documents.

    Returns:
        np.ndarray: TF-IDF feature matrix (dense).
    """
    print("   - Generating TF-IDF features...")

    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(texts)  # returns sparse matrix

    X_dense = X_tfidf.toarray() # Convert to dense NumPy array as requested

    # The 'vectorizer' object (CountVectorizer instance) is returned here
    # because it is 'fitted' on the training data and contains the learned
    # vocabulary. This fitted object is essential to consistently transform
    # new, unseen data into the same feature space.
    
    return X_dense, vectorizer