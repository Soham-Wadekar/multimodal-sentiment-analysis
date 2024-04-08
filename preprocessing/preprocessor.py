import os
from dotenv import load_dotenv, find_dotenv
from utils.text_utils import *

# For text preprocessing
import re

extracted_file_path = os.environ.get("EXTRACTED_FILE_PATH")
preprocessed_file_path = os.environ.get("PREPROCESSING_FILE_PATH")

def preprocess_audio():
    ...

def preprocess_videoframes():
    ...

def preprocess_text(text):

    # Remove usernames
    text = re.sub(r"@\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Lowercasing
    text = text.lower()

    # Removing chat words (abbreviations used while chatting)
    for chtwrd, meaning in chat_words.items():
        text = text.replace(chtwrd, meaning)

    # Removing punctuation
    text = "".join(ch for ch in text if ch not in punct_words)

    # Removing stopwords
    text = " ".join(word for word in text.split() if word not in stop_words)

    # Removing special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    return text