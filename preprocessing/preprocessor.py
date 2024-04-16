import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# For text preprocessing
import re
from utils.text_utils import *

# For audio preprocessing
import librosa
import pandas as pd
import numpy as np
from utils.audio_utils import *

extracted_file_path = os.environ.get("EXTRACTED_FILE_PATH")
preprocessed_file_path = os.environ.get("PREPROCESSING_FILE_PATH")

def preprocess_audio(input_path):
    
    features, labels = [], []

    for filename in os.listdir(input_path):
        
        # Adding labels
        sentiment = filename[6:8]
        if sentiment in ['04', '05', '07']:
            labels.append(-1)
        elif sentiment in ['01']:
            labels.append(0)
        elif sentiment in ['02', '03', '08']:
            labels.append(1)

        # Load the data
        y, sr = librosa.load(os.path.join(input_path, filename), sr=16000)

        # Normalizing the audio file
        y_normalized = librosa.util.normalize(y)

        # Extracting Mel Frequency Cepstral Coeffients (MFCCs)
        mfccs = librosa.feature.mfcc(y=y_normalized, sr=sr, n_mfcc=13)

        # Scaling the MFCCs
        mfccs_scaled = (mfccs - np.mean(mfccs, axis=0)) / np.std(mfccs, axis=0)

        # Making the 2D array into 1D
        mfccs_flattened = mfccs_scaled.flatten()

        features.append(mfccs_flattened)

    # Creating the data
    data = pd.DataFrame({
        'Name': filename,
        'Features': features,
        'Label': labels
    })

    # Padding all the data points for uniformity
    max_len = data["Features"].apply(len).max()
    data['Features'] = data['Features'].apply(lambda x: pad_array(x, max_len))

    data['Features'] = data['Features'].apply(lambda x: np.reshape(x, (165, 13)))

    return data


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