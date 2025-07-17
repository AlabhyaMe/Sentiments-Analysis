# Sentiment Analysis Pipeline

This repository hosts an easy-to-use, ready-made Sentiment Analysis pipeline designed to get you started quickly with classifying text data. Everything you need, from data preprocessing to model training and prediction, is set up and configured.

## ✨ Features

End-to-End Pipeline: Go from raw text to sentiment predictions with minimal setup.

Automated Preprocessing: Includes robust text cleaning (lemmatization, stop word removal, punctuation handling, URL/emoji/HTML removal, etc.).

Multiple Text Representation Methods: Experiment with different ways to convert text into numerical features:

Bag-of-Words (BoW)

Term Frequency (TF)

TF-IDF (Term Frequency-Inverse Document Frequency)

Word Embeddings (Word2Vec - pre-trained Google News 300-dim model used)

Multiple Machine Learning Algorithms: Choose from powerful classification models:

Logistic Regression

Random Forest

XGBoost

Hyperparameter Tuned Models: All included machine learning models are configured with GridSearchCV for optimal performance, ensuring they find the best hyperparameters.

Modular Design: Code is organized into separate, clean modules for easy understanding and maintenance.

Prediction on New Data: Easily apply your trained model to new, unseen text data.

## 🚀 Getting Started
Follow these steps to get your sentiment analysis pipeline up and running:

### 1. Prerequisites

Git: For cloning the repository.

Python 3.8+: (Recommended to use Anaconda for environment management).

Anaconda/Miniconda: Highly recommended for managing Python environments and dependencies.

### 2. Clone the Repository

Open your terminal or Git Bash and run:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name # Navigate into the cloned directory

(Replace https://github.com/your-username/your-repo-name.git with the actual URL of your GitHub repository.)

### 3. Set Up Your Environment & Install Dependencies

It's highly recommended to create a dedicated Conda environment to avoid conflicts:

conda create -n sentiment_env python=3.9 # Create a new environment
conda activate sentiment_env             # Activate the environment
pip install -r requirements.txt        # Install all required libraries

(The requirements.txt file is located in the root of this repository and lists all necessary packages like polars, scikit-learn, gensim, xgboost, nltk.)

### 4. Download NLTK Data

NLTK requires some data files. Run the following Python commands once to download them:

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

### 5. Prepare Your Data

Your project expects data in specific locations:

Training Data: Place your training CSV file (e.g., train.csv) inside the Training Data/ directory.

It must contain a column named Response for the raw text and a column named Sentiment for the corresponding labels (e.g., "Positive", "Negative", "Neutral").

New Data for Prediction: Place your new CSV file (e.g., new_texts.csv) that you want to predict on inside the New Data/ directory.

It must contain a column named RawTextColumn (or whatever you configure in the notebook) for the raw text.

### 6. Run the Main Pipeline
All the required instructions and code to run the sentiment analysis pipeline are within the sentiment_analysis_main.ipynb Jupyter Notebook.

Launch Jupyter Lab/Notebook from your project's root directory:

jupyter lab
or
jupyter notebook

Open the sentiment_analysis_main.ipynb notebook.

Follow the instructions and execute the cells sequentially. The notebook guides you through data loading, preprocessing, model training, and making predictions on new data.

## ⚙️ Project Structure & Components
The core logic of this pipeline is organized into modular folders:

preprocessing.py: Contains all the text cleaning and preprocessing functions (e.g., pre_process).

Vect/: Houses different text representation (vectorization) methods. Each file contains a vectorize function:

bag_of_words_vectorizer.py

tfidf_vectorizer.py

word_embedding_vectorizer.py (Utilizes the pre-trained word2vec-google-news-300 model for rich embeddings.)

MLAlgo/: Contains the machine learning model training and prediction logic. Each file has a train_and_predict function:

logistic_regression_model.py

random_forest_model.py

xgboost_model.py

## ⚠️ Important Note on Code Modification
This pipeline is designed for ease of use. The machine learning algorithms (Logistic Regression, Random Forest, XGBoost) and text representation methods (Bag-of-Words, TF-IDF, Word Embeddings) are pre-configured with hyperparameter tuning.

If you know what you are doing and understand the implications of hyperparameter changes or algorithm modifications, you may tweak the code within the Vect/ and MLAlgo/ folders.

Otherwise, it is strongly recommended NOT to make any changes to these internal .py files, as incorrect modifications can lead to program crashes or unexpected behavior.

Enjoy your sentiment analysis journey!
