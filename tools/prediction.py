# predict_new_data.py

import polars as pl
import numpy as np
from sklearn.preprocessing import LabelEncoder # Needed for inverse_transform
import importlib

# Import your preprocessing function (assuming it's in preprocessing.py)
from tools.preprocessing import pre_process

# Import the specific sentence_vector if using word_embedding
# This is needed because your 'wv' vectorize function internally calls it.
# It's best to import it here directly if it's a helper for this prediction logic.
try:
    from Vect.wv import sentence_vector
except ImportError:
    # If you're sure you won't use 'wv' prediction, this can be removed.
    # Otherwise, ensure Vect.word_embedding_vectorizer.py exists and exposes sentence_vector.
    sentence_vector = None


def predictions(
    df_new: pl.DataFrame,
    raw_text_column_name: str, # Column with raw text in new DataFrame
    trained_model,
    vectorizer_object_from_training, # The fitted vectorizer/embedding model
    label_encoder_from_training: LabelEncoder,
    vectorizer_name_used: str, # Name of the vectorizer used during training
    pre_process_options_for_new_data: dict = None # Options to apply to new raw text
) -> pl.DataFrame:
    """
    Makes predictions on a new Polars DataFrame using a previously trained model and vectorizer.

    Args:
        df_new (pl.DataFrame): The new DataFrame containing raw text data to predict on.
        raw_text_column_name (str): The name of the column in df_new containing the raw text.
        trained_model: The scikit-learn model object that was previously trained.
        vectorizer_object_from_training: The fitted vectorizer (CountVectorizer, TfidfVectorizer)
                                         or the loaded Word2Vec model that was used during training.
        label_encoder_from_training (LabelEncoder): The fitted LabelEncoder used during training.
        vectorizer_name_used (str): The name of the vectorizer method used during training (e.g., 'tfidf', 'wv').
        pre_process_options_for_new_data (dict, optional): Dictionary of options to pass to the pre_process function.
                                                            MUST be the SAME as used for training data. Defaults to None.

    Returns:
        pl.DataFrame: The new DataFrame with a 'predicted_label' column.
    """
    if pre_process_options_for_new_data is None:
        pre_process_options_for_new_data = {}

    print(f"\n--- Predicting on New Data ---")
    print(f"Using vectorizer: {vectorizer_name_used.replace('_', ' ').title()}")

    # 1. Preprocess the new raw text data (MUST use the same options as training)
    print("1. Preprocessing new text data...")
    X_new_raw_texts = df_new[raw_text_column_name].to_list()

    # Determine return_string based on the vectorizer type used during training
    if vectorizer_name_used == 'wv':
        # Word2Vec vectorizer expects list of lists (tokens)
        X_new_processed = [pre_process(doc, **pre_process_options_for_new_data, return_string=False) for doc in X_new_raw_texts]
    else:
        # TFIDF/BoW vectorizers expect list of strings
        X_new_processed = [pre_process(doc, **pre_process_options_for_new_data, return_string=True) for doc in X_new_raw_texts]


    # 2. Vectorize the new preprocessed text using the *trained* vectorizer
    print("2. Vectorizing new text data...")
    if vectorizer_name_used == 'wv':
        if sentence_vector is None:
            raise ImportError("sentence_vector function not found. Ensure it's accessible for 'wv' vectorizer.")
        # For Word2Vec, apply sentence_vector using the loaded gensim model
        X_new_features = np.array([
            sentence_vector(tokens, vectorizer_object_from_training)
            for tokens in X_new_processed
        ])
    else:
        # For CountVectorizer/TfidfVectorizer, use .transform()
        X_new_features = vectorizer_object_from_training.transform(X_new_processed)

    # 3. Make predictions using the trained model
    print("3. Making predictions...")
    new_predictions_numerical = trained_model.predict(X_new_features)

    # 4. Inverse transform predictions to original labels
    new_predictions_original_labels = label_encoder_from_training.inverse_transform(new_predictions_numerical)

    # 5. Add predictions to the new DataFrame
    df_new_with_predictions = df_new.with_columns(
        pl.Series("predicted_label", new_predictions_original_labels)
    )

    print("Predictions complete.")
    return df_new_with_predictions