
# POSITIVELY DO NOT CHANGE

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import polars as pl
import importlib
import numpy as np

def run_pipeline(vectorizer_name, model_name, df: pl.DataFrame):
    """
    Runs the full pipeline:
      - vectorize
      - train model
      - evaluate
    """
    # do not change the code below
    # Import vectorizer from Vect folder
    vec_module = importlib.import_module(f"Vect.{vectorizer_name}")
    vectorizer_func = getattr(vec_module, "vectorize")

    # Import ML model from MLAlgo folder
    model_module = importlib.import_module(f"MLAlgo.{model_name}")
    model_func = getattr(model_module, "train_and_predict")

    # Prepare data
    X_text = df["processed"].to_list()
    y_raw = df["Sentiment"].to_list()

    #XGBoost need Label Encoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    

    # Vectorize
    X = vectorizer_func(X_text) #passes the processed text to the vectorizer function, which returns the vectorized representation
                                # it is processed in the respective vectorizer function, which is imported from the Vect folder

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train + predict
    y_pred, ml_model = model_func(X_train, y_train, X_test)

    # Evaluate
    print(classification_report(y_test, y_pred))
    print(f"Labels encoded: Original -> {label_encoder.classes_}, Encoded -> {np.unique(y)}")
    print("True labels distribution:", Counter(y_test))
    print("Predicted labels distribution:", Counter(y_pred))  

    new_predictions_original_labels = label_encoder.inverse_transform(y_pred)

    return X , vectorizer_func, ml_model