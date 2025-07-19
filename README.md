# 💬 Sentiment Analysis Pipeline

This repository hosts an easy-to-use, ready-made **Sentiment Analysis pipeline** designed to get you started quickly with classifying text data. Everything you need, from data preprocessing to model training and prediction, is set up and configured.

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
git clone https://github.com/AlabhyaMe/Sentimental-Analysis-.git
cd Sentimental-Analysis-
conda create -n sentiment_env python=3.9
conda activate sentiment_env
pip install -r requirements.txt
```
```
This project is setup in the follwing pipeline
├── Training Data/
│   └── train.csv                # Your training file
├── New Data/
│   └── new_texts.csv            # Your new prediction file
├── MLAlgo/
│   ├── logistic_regression_model.py
│   ├── random_forest_model.py
│   └── xgboost_model.py
├── Vect/
│   ├── bag_of_words_vectorizer.py
│   ├── tfidf_vectorizer.py
│   └── word_embedding_vectorizer.py
├── preprocessing.py             # Text cleaning functions
├── sentiment_analysis_main.ipynb  # Full training + prediction notebook
├── sentiment_prediction.ipynb     # Quick prediction-only notebook
├── requirements.txt
└── README.md


```


## 3. Prepare Your Data

### 📌 Training Data

Place your training CSV file in the `Training Data/` folder.

- It must contain:
  - A column named `Response` – for the raw input text.
  - A column named `Sentiment` – for sentiment labels (e.g., `"Positive"`, `"Negative"`, `"Neutral"`).

### 📌 New Data for Prediction

Place your new prediction CSV file in the `New Data/` folder.

- It must contain:
  - A column named `RawTextColumn` (or another name you configure in the notebook).

## 📚 Dataset Citation

This project uses publicly available training data from:

> Madhav Kumar Choudhary. *Sentiment Prediction on Movie Reviews*. Kaggle.  
> [https://www.kaggle.com/datasets/madhavkumarchoudhary/sentiment-prediction-on-movie-reviews](https://www.kaggle.com/datasets/madhavkumarchoudhary/sentiment-prediction-on-movie-reviews)  
> Accessed on: 2025- 07-15

If you use this dataset in your own work, please cite the original creator as per Kaggle's [Terms of Use](https://www.kaggle.com/terms).

