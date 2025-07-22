#  Quick Sentiments

## Updates
The package is now available to use. 

```bash
pip install -i https://test.pypi.org/simple/ quick-sentiments
```
Due to Test PyPl not having the latest dependencies, it is recommended that you clone the Git repository and install the package locally. 

```bash
git clone https://github.com/AlabhyaMe/quick_sentiments.git
```
Then run the command in the command prompt or notebook where git is cloned. Make sure you are in the main directory - quick_sentiments

```
pip install .\dist\quick_sentiments-0.1.8-py3-none-any.whl
```

This Python package is designed to streamline natural language processing (NLP) for sentiment analysis. It achieves this by combining various vectorization techniques with machine learning models. The package automates the often complex and time-consuming vectorization process, allowing users to skip the manual coding typically required for this step. Additionally, users can easily select their preferred machine learning models to conduct sentiment analysis.


##  Features

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
##  Package Structure 
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


## 3. USE THE DEMO

The DEMO has workbooks that are ready to use. Just ensure that your files and column names are correctly labelled. Follow the instructions in the workbook. Alternatively, you can just run the Python script if files and labels are properly set. 

###  Training Data

Place your training CSV file in the `demo/training_data` folder.

- It must contain:
  - A column for  the raw input text. 
  - A column for sentiments

### New Data for Prediction

Place your new prediction CSV file in the `new_data/` folder.

- It must contain:
  - A column named `RawTextColumn` (or another name you configure in the notebook).

## 📚 Dataset Citation

The demo uses publicly available training data from:

> Madhav Kumar Choudhary. *Sentiment Prediction on Movie Reviews*. Kaggle.  
> [https://www.kaggle.com/datasets/madhavkumarchoudhary/sentiment-prediction-on-movie-reviews](https://www.kaggle.com/datasets/madhavkumarchoudhary/sentiment-prediction-on-movie-reviews)  
> Accessed on: 2025- 07-15

If you use this dataset in your own work, please cite the original creator as per Kaggle's [Terms of Use](https://www.kaggle.com/terms).

