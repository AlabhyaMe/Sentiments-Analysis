from typing import Union
import polars as pl
import pandas as pd
import numpy as np

def make_predictions(
        new_data: Union[pl.DataFrame, pd.DataFrame],
        text_column_name: str,
        vectorizer,
        best_model,
        label_encoder,
        prediction_column_name: str = "predictions") -> pl.DataFrame:
    """
    Makes predictions and adds them as a new column with original labels.
    
    Args:
        new_data: Input DataFrame (Polars or pandas)
        text_column_name: Name of column containing text to predict on
        vectorizer: Fitted vectorizer (TF-IDF/BOW) or word embeddings model
        best_model: Trained model (must have classes_ attribute)
        label_encoder: Fitted LabelEncoder for inverse transform
        prediction_column_name: Name for new prediction column
        
    Returns:
        Polars DataFrame with label predictions added
    """
    # Convert pandas to Polars if needed
    if isinstance(new_data, pd.DataFrame):
        new_data = pl.from_pandas(new_data)
    elif not isinstance(new_data, pl.DataFrame):
        raise TypeError(f"Expected Polars or pandas DataFrame, got {type(new_data)}")

    # Drop nulls in the text column
    new_data = new_data.drop_nulls(subset=[text_column_name])
    texts = new_data[text_column_name].to_list()
    
    # Generate features
    if hasattr(vectorizer, 'transform'):
        new_features = vectorizer.transform(texts)
    else:
        def text_to_vector(text):
            words = text.split()
            vectors = [vectorizer[word] for word in words if word in vectorizer]
            return np.mean(vectors, axis=0) if vectors else np.zeros(vectorizer.vector_size)
        new_features = np.array([text_to_vector(text) for text in texts])
    
    # Get numerical predictions
    numeric_predictions = best_model.predict(new_features)
    
    # Convert to original labels
    predictions = label_encoder.inverse_transform(numeric_predictions)
    
    # Add predictions as new column
    return new_data.with_columns(
        pl.Series(prediction_column_name, predictions)
    )