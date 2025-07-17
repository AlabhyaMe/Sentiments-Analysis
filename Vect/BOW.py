
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np # Although CountVectorizer returns sparse, it's good practice to import np

def vectorize(texts):
    """
    Generates Bag-of-Words (BoW) features for the entire dataset.

    Args:
        texts (list[str]): List of all preprocessed documents (strings).
                           Each string is expected to be a cleaned document (e.g., "word1 word2").

    Returns:
        tuple: (X_features, vectorizer_object)
               X_features (scipy.sparse.csr_matrix): The Bag-of-Words matrix.
               vectorizer_object (sklearn.feature_extraction.text.CountVectorizer):
                   The fitted CountVectorizer object.
    """
    print("   - Generating Bag-of-Words features...")
    vectorizer = CountVectorizer()
    # Fit and transform on the entire dataset as per the current run_pipeline design
    X_features = vectorizer.fit_transform(texts)

    return X_features