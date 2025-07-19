# quick_sentiments/__init__.py
from .pipeline import run_pipeline       # Expose pipeline function
from .prediction import make_predictions # Expose prediction function

__all__ = ['run_pipeline', 'make_predictions']  # Controls what's available in *