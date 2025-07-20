# 💬 Sentiment Analysis Pipeline

This repository hosts an easy-to-use, ready-made **Sentiment Analysis pipeline** designed to get you started quickly with classifying text data. Everything you need, from data preprocessing to model training and prediction, is set up and configured.

Find the notebook DEMO sentiment prediction.ipynb that would walk through the methods. Alternatively, if you only want to run through  the vectorization and ML predictions, run the  sentiment_prediction notebook or Python file. 

Just clone the repository to get started

```bash
git clone https://github.com/AlabhyaMe/quick_sentiments-.git
```

---

## ✨ Features

- **End-to-End Pipeline**: Go from raw text to sentiment predictions with minimal setup.
- **Automated Preprocessing**: Includes robust text cleaning:
  - Lemmatization
  - Stop word removal
  - Punctuation handling
  - URL/emoji/HTML removal, etc.
- **Multiple Text Representation Methods**:
  - Bag-of-Words (BoW)
  - Term Frequency (TF)
  - TF-IDF (Term Frequency-Inverse Document Frequency)
  - Word Embeddings (Word2Vec - pre-trained Google News 300-dim model)
- **Multiple Machine Learning Algorithms**:
  - Logistic Regression
  - Random Forest
  - XGBoost
- **Hyperparameter Tuning Support**:
  - All models are compatible with GridSearchCV.
  - By default, models run with standard parameters for quick testing.
  - Grid search options are built-in and ready to use if needed.
- **Modular Design**: Each component is cleanly separated into its own module.
- **Prediction on New Data**: Easily apply your trained model to new, unseen data.

---

## 🚀 Getting Started

Follow these steps to get your sentiment analysis pipeline up and running:

### 1. Prerequisites

- **Git**: For cloning the repository.
- **Python 3.8+** (Recommended: Anaconda for environment management)
- **Anaconda/Miniconda**: Strongly recommended

### 2. Clone the Repository

```bash
git clone https://github.com/AlabhyaMe/quick_sentiments-.git
cd quick_sentiments
conda create -n sentiment_env python=3.9
conda activate sentiment_env
pip install -r requirements.txt
```
```
quick_sentiments/                
├── quick_sentiments/            
│   ├── demo/                   
│   │   └── new_data  #new file for predictions to be stored here | predicted files will also be generated here
                |-test.csv  
        └── training_data
                └── train.csv # demo train data is here
│   ├── ml_algo/                 
│   │   ├── __init__.py          
│   │   ├── logt.py
│   │   ├── rf.py
│   │   └── XGB.py
│   ├── vect/                    
│   │   ├── __init__.py          
│   │   ├── BOW.py
│   │   ├── tf.py
        ├── tfidf.py
│   │   └── wv.py
    ├── DEMO sentiment_prediction.ipynb   # demo of how to use the notebook
│   ├── pipeline.py             
│   ├── predict.py               
│   ├── preprocess.py            
│   ├── sentiment_prediction.ipynb  # can be used by the user to make prediction
│   ├── sentiment_prediction.py  # A standalone Python script for prediction 
│   └── virtual environment setup.py 
│                                
├── README.md                    # Project description and instructions
├── requirements.txt             # All Python dependencies
├── setup.py                     # For optional future packaging (top-level)
├── tests/                       # Your test files
├── dist/                        # Built package distributions (automatically generated)
└── pyproject.toml.txt           # NEW: This is likely pyproject.toml with a wrong .txt extension. It should just be `pyproject.toml`.

```


## 3. Prepare Your Data

### 📌 Training Data

Place your training CSV file in the `demo/training_data` folder.

- It must contain:
  - A column for  the raw input text. 
  - A column for sentiment labels (e.g., `"Positive"`, `"Negative"`, `"Neutral"`)

### 📌 New Data for Prediction

Place your new prediction CSV file in the `new_data/` folder.

- It must contain:
  - A column named `RawTextColumn` (or another name you configure in the notebook).

## 📚 Dataset Citation

The demo uses publicly available training data from:

> Madhav Kumar Choudhary. *Sentiment Prediction on Movie Reviews*. Kaggle.  
> [https://www.kaggle.com/datasets/madhavkumarchoudhary/sentiment-prediction-on-movie-reviews](https://www.kaggle.com/datasets/madhavkumarchoudhary/sentiment-prediction-on-movie-reviews)  
> Accessed on: 2025- 07-15

If you use this dataset in your own work, please cite the original creator as per Kaggle's [Terms of Use](https://www.kaggle.com/terms).

