import pytest
from quick_sentiments import preprocess

def test_clean_text():
    assert preprocess.clean_text("Hello World! 123") == "hello world "
    assert preprocess.clean_text("Test. With, Punctuation?") == "test with punctuation"

def test_tokenize_text():
    assert preprocess.tokenize_text("hello world") == ["hello", "world"]

def test_remove_stopwords():
    tokens = ["this", "is", "a", "test", "sentence"]
    # Assuming 'is', 'a' are stopwords
    assert preprocess.remove_stopwords(tokens) == ["test", "sentence"]