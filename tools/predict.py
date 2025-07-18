import importlib

# this is for new data
# we do not expect sentiment_column_name in the new data
# if there is sentiment_column_name, user should calculate the accuracy and F1 score manually
def predict_pipeline(
        df,
        text_column_name: str,
        vectorizer_func,
        ml_model):
    """
    Predicts the sentiment of new data using the provided vectorizer and model functions.
    
    Parameters:
    - new_data: DataFrame containing the new data to predict.
    - vectorizer_func: Function to vectorize the text data.
    - model_func: Function to apply the trained model for prediction.
    
    Returns:
    - predictions: Array of predicted sentiments.
    """
    try:
        vec_module = importlib.import_module(f"Vect.{vectorizer_func}")
        vectorize_function = getattr(vec_module, "vectorize")
    except (ImportError, AttributeError) as e:
        print(f"Error loading vectorizer module/function: {e}")
        return None   
        # Prepare data

    X_text = df[text_column_name].to_list() 
    # Vectorize the new data
    X_new,vect_method = vectorize_function(X_text)  # Passes the processed text to the vectorizer function, which returns the vectorized representation
    #since i am reusing the vectorizer function, it will return the vectorized data in the same format as before, which is not necessary anymore

    # This is new data, so we do not split it into train and test sets
    # Instead, we directly use the vectorized data for prediction
    # Note, we are not training the model here, and these data should not be used for training to avoid data leakage
    
    # Predict using the model
    predictions = ml_model.predict(X_new)
    
    
    return  predictions