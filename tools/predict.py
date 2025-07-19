import polars as pl
def make_predictions(
        new_data: pl.DataFrame,
        text_column_name: str,
        vectorizer,
        best_model,
        label_encoder,
        prediction_column_name: str = "predictions") -> pl.DataFrame:
    """
    Makes predictions and adds them as a new column with original labels.
    
    Args:
        new_data: Input Polars DataFrame
        text_column_name: Name of column containing text to predict on
        vectorizer: Fitted vectorizer (TF-IDF/BOW) or word embeddings model
        best_model: Trained model (must have classes_ attribute)
        prediction_column_name: Name for new prediction column
        
    Returns:
        Polars DataFrame with label predictions added
    """
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
    
    # Get numerical predictions and map to labels
    numeric_predictions = best_model.predict(new_features)
    
    # Check if model has label mapping
    if hasattr(best_model, 'classes_'):
        label_map = best_model.classes_
        predictions = [label_map[pred] for pred in numeric_predictions]
    else:
        predictions = numeric_predictions  # fallback to numeric if no mapping
    
    predictions = label_encoder.inverse_transform(numeric_predictions)
    
    # Add predictions as new column
    return new_data.with_columns(
        pl.Series(prediction_column_name, predictions)
    )