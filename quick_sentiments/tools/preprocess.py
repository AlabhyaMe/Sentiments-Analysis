import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import unicodedata

# This block ensures NLTK data is downloaded if not found.
# It's good practice to have this check in a module that relies on NLTK data.
# --- NLTK Downloads (Run these once, preferably outside the script or with a check) ---
# This block ensures NLTK data is downloaded if not found.
# It's good practice to have this check in a module that relies on NLTK data.
def _download_nltk_data():
    """Helper function to download NLTK data if not present."""
    # List of NLTK datasets required by this module
    # 'punkt' for word_tokenize, 'stopwords' for stopwords, 'wordnet' for WordNetLemmatizer
    # 'omw-1.4' is often recommended for WordNetLemmatizer for broader coverage.
    datasets = ['punkt', 'stopwords', 'wordnet', 'omw-1.4'] # <--- ADDED 'omw-1.4'

    for dataset in datasets:
        try:
            # NLTK data paths vary, 'punkt' is a tokenizer, others are corpora.
            # This check tries to find it based on common NLTK data structures.
            if dataset == 'punkt':
                nltk.data.find(f'tokenizers/{dataset}')
            elif dataset == 'omw-1.4': # omw-1.4 is part of corpora
                nltk.data.find(f'corpora/{dataset}')
            else: # stopwords, wordnet are also in corpora
                nltk.data.find(f'corpora/{dataset}')
            print(f"NLTK data '{dataset}' already present.")
        except Exception as e: # Catch any exception if data is not found
            print(f"Downloading NLTK data: {dataset}...")
            nltk.download(dataset)
            print(f"NLTK data '{dataset}' downloaded.")
        # Note regarding 'punkt_tab': There isn't a standard 'punkt_tab' dataset.
        # It's likely a typo for 'punkt'. 'punkt' is already included.

_download_nltk_data() # Call this helper function when the module is imported

# --- Initialize NLTK components once ---
# These are global to this module
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- Initialize NLTK components once ---
# These are global to this module
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))



# --- Helper Cleaning Functions ---
def remove_square_brackets(text):
    """Removes text enclosed in square brackets."""
    cleaned_text = re.sub(r'\[.*?\]', '', text)
    return cleaned_text.strip()

def remove_urls_emails(text):
    """Removes URLs and email addresses from text."""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S*@\S*\s?', '', text)
    return text

def remove_html_tags(text):
    """Removes HTML tags from text."""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_numbers(text):
    """Removes all digits from text."""
    return re.sub(r'\d+', '', text)

def remove_extra_spaces(text):
    """Replaces multiple spaces with a single space and strips leading/trailing spaces."""
    return re.sub(r'\s+', ' ', text).strip()

def remove_punctuation_from_token(token):
    """Removes punctuation characters from an individual token."""
    translator = str.maketrans('', '', string.punctuation)
    return token.translate(translator)

def remove_emojis(text):
    """Removes common emojis from text using a regex pattern."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def normalize_unicode_characters(text): # <--- NEW HELPER FUNCTION
    """
    Normalizes unicode characters (e.g., smart quotes, accented chars)
    to their closest ASCII equivalents and removes non-ASCII.
    """
    # Normalize to NFKD form (decomposes characters like é to e + accent)
    # Then encode to ASCII, ignoring characters that can't be mapped
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')


# --- Main Preprocessing Function with Options ---
def pre_process(doc,
                remove_brackets=True,
                remove_urls=True,
                remove_html=True,
                remove_nums=False,
                remove_emojis_flag=False,
                normalize_unicode=True,
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
        normalize_unicode (bool): If True, normalize unicode characters to ASCII.
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
    if remove_emojis_flag:
        doc = remove_emojis(doc)
    if normalize_unicode: # <--- NEW STEP IN PIPELINE
        doc = normalize_unicode_characters(doc)
    if to_lowercase:
        doc = doc.lower()
    if remove_extra_space:
        doc = remove_extra_spaces(doc)

    # Stage 2: Tokenization
    tokens = []
    if tokenize:
        tokens = word_tokenize(doc)
    else:
        # If not tokenizing, just return the string after initial cleaning
        # Ensure it's a list if return_string is False, even if it's a single item
        return [doc.strip()] if not return_string else doc.strip()

    # Stage 3: Token-level cleaning and normalization
    processed_tokens = []
    for token in tokens:
        if remove_punct_tokens:
            token = remove_punctuation_from_token(token)

        if not token: # Skip if token became empty after cleaning punctuation
            continue

        if remove_stop_words:
            if token in stop_words:
                continue

        if lemmatize:
            token = lemmatizer.lemmatize(token)

        if token: # Final check in case lemmatization or other steps create empty strings
            processed_tokens.append(token)

    if return_string:
        return " ".join(processed_tokens)
    else:
        return processed_tokens