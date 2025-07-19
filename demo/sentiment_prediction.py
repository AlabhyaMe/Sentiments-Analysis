#load the packages
import polars as pl
#make sure that this is the main file
import sys
import os
project_root = os.getcwd()
sys.path.insert(0, project_root)
# here I have three python script I built to pre_process the data and running the pipeline
# you can find the code in the tools/preprocess.py file
# you can find  the code in the tools/pipeline.py file
# the pre_process function is used to clean the text data, there are various options available, please check the tools/preprocess.py file for details
# the run_pipeline function is used to run the sentimental analysis pipeline, it takes the training data and the vectorizer and machine learning methods as input, and returns the results
from tools.preprocess import pre_process
#this function will run the sentimental analysis in the training data and return the results
from tools.pipeline import run_pipeline
# this function will run the sentimental analysis in the new data and return the predictions
from tools.predict import make_predictions

# ENTER YOUR PATHS HERE FOR THE TRAINING DATA SET
path1 = "Training Data/Train.csv" #give path to the training data
df_train = pl.read_csv(path1, has_header=True, encoding="utf8")

# ENTER YOUR PATHS HERE FOR THE TESTING DATA SET
path2 = "New Data/Test.csv" #give path to the test data
df_test = pl.read_csv(path2, has_header=True, encoding="utf8")


# ENTER THE COLUMN NAMES HERE
response_column = "reviewText" # feel free to change the column name to your text column name
sentiment_column = "sentiment" # feel free to change the column name to your label column name   

df_train = df_train.with_columns(
    pl.col(response_column).map_elements(lambda x: pre_process(x)).alias("processed")  #add inside the map_elements
)

dt= run_pipeline(
    vectorizer_name="BOW", # BOW, tf, tfidf, wv
    model_name="logit", # logit, rf, XGB .#XGB takes long time, can not recommend using it on normal case
    df=df_train,
    text_column_name="processed",  # this is the column name of the text data, 
    sentiment_column_name = "sentiment",
    perform_tuning = False # make this true if you want to perform hyperparameter tuning, it will take longer time and 
                            # may run out of memory if the dataset is large,
)

new_data = df_test.with_columns(
    pl.col(response_column).map_elements(lambda x: pre_process(x).alias("processed")  #add inside the map_elements
)

# MAKE PREDICTIONS ON THE NEW DATA
sentiments_prediction= make_predictions(
    new_data=new_data,
    text_column_name="processed",
    vectorizer=dt["vectorizer_object"],
    best_model=dt["model_object"],
    label_encoder=dt["label_encoder"],
    prediction_column_name="sentiment_predictions"  # Optional custom name
)

# SAVE THE PREDICTIONS TO A CSV FILE
sentiments_prediction.write_csv("New Data/sentiments_prediction.csv")

# THE END