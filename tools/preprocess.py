#Do not change the code
#these are the list of various functions you can use. you don't need to make any changes. 
#in the next instructions, I will present you the options that you can choose for your text cleaning
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy
def remove_punctuation_from_token(token):
    """
    Removes punctuation characters from an individual token.
    Uses string.punctuation for a comprehensive list of punctuation.
    """
    # Create a translation table: maps each punctuation character to None (meaning delete it)
    translator = str.maketrans('', '', string.punctuation)
    # Apply the translation to the token
    return token.translate(translator)

def remove_square_brackets(text):
    # This pattern matches any substring inside square brackets, including the brackets
    cleaned_text = re.sub(r'\[.*?\]', '', text)
    return cleaned_text.strip()

def simple_tokenizer(text):
    # Basic word tokenizer: split on non-word characters
    return re.findall(r'\b\w+\b', text)

def remove_urls_emails(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove email addresses
    text = re.sub(r'\S*@\S*\s?', '', text)
    return text

def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_numbers(text):
    return re.sub(r'\d+', '', text) # Removes all digits
#Caution, numbers may be cruitial in finance and economics

def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip() # Replaces multiple spaces with one, then trims leading/trailing

# To remove emojis (example, requires `emoji` package for comprehensive removal)
import emoji
def remove_emojis(text):
    return emoji.get_emoji_regexp().sub(r'', text)

# Do not change the code
# -- Main Preprocessing Function with Options ---
#have a careful look at this function 
#this function cleans the data using the functions from above
#some functions are set to False by defualt, if you need them, set it to true (not in this snippet of code)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def pre_process(doc,
                remove_brackets=False,
                remove_urls=True,
                remove_html=True,
                remove_nums=False,
                remove_emojis_flag=False, 
                to_lowercase=True,
                tokenize=True,
                remove_punct_tokens=True,
                remove_stop_words=True,
                lemmatize=True,
                remove_extra_space=True,
                return_string=True):
    """
    Preprocesses a text document with configurable cleaning steps.

    Args:
        doc (str): The input text document.
        remove_brackets (bool): If True, remove text in square brackets.
        remove_urls (bool): If True, remove URLs and email addresses.
        remove_html (bool): If True, remove HTML tags.
        remove_nums (bool): If True, remove all numeric digits.
        remove_emojis_flag (bool): If True, remove common emojis.
        to_lowercase (bool): If True, convert text to lowercase.
        tokenize (bool): If True, tokenize the text using NLTK's word_tokenize.
        remove_punct_tokens (bool): If True, remove punctuation from individual tokens.
        remove_stop_words (bool): If True, remove common English stop words.
        lemmatize (bool): If True, perform lemmatization on tokens.
        remove_extra_space (bool): If True, replace multiple spaces with single spaces.
        return_string (bool): If True, join tokens back into a string; otherwise, return a list of tokens.

    Returns:
        str or list: The preprocessed text as a string or a list of tokens.
    """

    # Stage 1: Text-level cleaning (before tokenization)
    if remove_brackets:
        doc = remove_square_brackets(doc)
    if remove_urls:
        doc = remove_urls_emails(doc)
    if remove_html:
        doc = remove_html_tags(doc)
    if remove_nums:
        doc = remove_numbers(doc)
    if remove_emojis_flag: # Integrated emoji removal
        doc = remove_emojis(doc)
    if to_lowercase:
        doc = doc.lower()
    if remove_extra_space:
        doc = remove_extra_spaces(doc)

    # Stage 2: Tokenization
    tokens = []
    if tokenize:
        tokens = word_tokenize(doc)
    else:
        return doc.strip() if return_string else [doc.strip()]

    # Stage 3: Token-level cleaning and normalization
    processed_tokens = []
    for token in tokens:
        if remove_punct_tokens:
            token = remove_punctuation_from_token(token)

        if not token:
            continue

        if remove_stop_words:
            if token in stop_words:
                continue

        if lemmatize:
            token = lemmatizer.lemmatize(token)

        if token:
            processed_tokens.append(token)

    if return_string:
        return " ".join(processed_tokens)
    else:
        return processed_tokens
    
print("tools.preprocess module loaded")

print("Functions available in module:", dir())
