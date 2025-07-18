# POSITIVELY DO NOT CHANGE (This comment is for the user, I am making necessary fixes based on our conversation)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import polars as pl
import importlib
import numpy as np

def run_pipeline(
    vectorizer_name: str,
    model_name: str,
    df: pl.DataFrame,
    text_column_name: str,       
    sentiment_column_name: str   
):
    """
    Runs the full pipeline:
      - vectorize
      - train model
      - evaluate

    Args:
        vectorizer_name (str): Name of the vectorizer (e.g., 'tfidf', 'word_embedding').
        model_name (str): Name of the ML model (e.g., 'logistic_regression', 'random_forest').
        df (pl.DataFrame): Your Polars DataFrame containing the text and sentiment columns.
        text_column_name (str): The name of the column in `df` that contains the processed text.
        sentiment_column_name (str): The name of the column in `df` that contains the sentiment labels.

    Returns:
        dict: A dictionary containing the trained model, fitted vectorizer, label encoder, and evaluation results.
    """
    print(f"--- Running Pipeline for {vectorizer_name.replace('_', ' ').title()} + {model_name.replace('_', ' ').title()} ---")

    # Import vectorizer from Vect folder
    try:
        vec_module = importlib.import_module(f"Vect.{vectorizer_name}")
        vectorize_function = getattr(vec_module, "vectorize")
    except (ImportError, AttributeError) as e:
        print(f"Error loading vectorizer module/function: {e}")
        return None

    # Import ML model from MLAlgo folder
    try:
        model_module = importlib.import_module(f"MLAlgo.{model_name}")
        train_and_predict_function = getattr(model_module, "train_and_predict")
    except (ImportError, AttributeError) as e:
        print(f"Error loading ML model module/function: {e}")
        return None

    # Prepare data using the provided column names
    X_text = df[text_column_name].to_list() 
    y_raw = df[sentiment_column_name].to_list() 

    # Label Encoding for y_raw
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    print(f"Labels encoded: Original -> {label_encoder.classes_}, Encoded -> {np.unique(y)}")

    # Vectorize the entire dataset (X)
    print("1. Vectorizing entire dataset (X)...")
    X_vectorized, fitted_vectorizer_object = vectorize_function(X_text)

    # Split data AFTER vectorization
    print("2. Splitting data into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train + predict
    print("3. Training and predicting...")
    y_pred, trained_model_object = train_and_predict_function(X_train, y_train, X_test)

    # Evaluate
    print("4. Evaluating model...")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("True labels distribution:", Counter(y_test))
    print("Predicted labels distribution:", Counter(y_pred))

    # Return results including all necessary objects for future predictions
    return {
        "model_object": trained_model_object,
        "vectorizer_name": vectorizer_name,
        "vectorizer_object": fitted_vectorizer_object,
        "label_encoder": label_encoder,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True, target_names=label_encoder.classes_)
    }