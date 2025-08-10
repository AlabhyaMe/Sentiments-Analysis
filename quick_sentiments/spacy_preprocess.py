import re
import unicodedata

# --- Global spaCy NLP object ---
nlp = None
spacy_available = False

try:
    import spacy
    spacy_available = True
    try:
        nlp = spacy.load("en_core_web_sm")
        print("SpaCy and model 'en_core_web_sm' loaded successfully.")
    except OSError:
        print("SpaCy model 'en_core_web_sm' not found. Attempting to download...")
        try:
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
            print("SpaCy model 'en_core_web_sm' downloaded and loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to download or load spaCy model: {e}")
            print("Please try downloading manually: python -m spacy download en_core_web_sm")
            nlp = None

except ImportError:
    # This will be triggered if 'pip install spacy' was not run.
    # We will raise an error within the function if tokenization is attempted.
    pass

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

def remove_extra_spaces(text):
    """Replaces multiple spaces with a single space and strips leading/trailing spaces."""
    return re.sub(r'\s+', ' ', text).strip()

def remove_emojis(text):
    """Removes common emojis from text using a regex pattern."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def normalize_unicode_characters(text):
    """
    Normalizes unicode characters to their closest ASCII equivalents and removes non-ASCII.
    """
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

# --- Main Preprocessing Function ---
def pre_process_spacy(doc,
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

    # Stage 1: Text-level cleaning
    if remove_brackets:
        doc = remove_square_brackets(doc)
    if remove_urls:
        doc = remove_urls_emails(doc)
    if remove_html:
        doc = remove_html_tags(doc)
    if remove_emojis_flag:
        doc = remove_emojis(doc)
    if normalize_unicode:
        doc = normalize_unicode_characters(doc)
    if to_lowercase:
        doc = doc.lower()
    if remove_extra_space:
        doc = remove_extra_spaces(doc)

    # Stage 2 & 3: Tokenization & Token-level cleaning
    if tokenize:
        # Crucial check: raise a specific error if spaCy is needed but not available
        if not spacy_available or nlp is None:
            raise RuntimeError(
                "Cannot perform tokenization. The 'spacy' library or its required model "
                "'en_core_web_sm' is not installed or failed to load. "
                "Please check the warnings printed during script initialization for instructions."
            )
        
        doc_spacy = nlp(doc)
        processed_tokens = []
        for token in doc_spacy:
            if token.is_space:
                continue
            if remove_punct_tokens and token.is_punct:
                continue
            if remove_stop_words and token.is_stop:
                continue
            if remove_nums and token.like_num:
                continue
            
            processed_token = token.lemma_ if lemmatize else token.text
            
            if processed_token.strip():
                processed_tokens.append(processed_token)
    else:
        return [doc.strip()] if not return_string else doc.strip()

    # Stage 4: Final output
    if return_string:
        return " ".join(processed_tokens).strip()
    else:
        return processed_tokens