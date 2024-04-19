import cv2
import numpy as np
import os
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

from extraction.extractor import extract
from preprocessing.preprocessor import *

import warnings
warnings.filterwarnings("ignore")

extract("video.mp4")

sentiment_score = {}

# Loading all the models

text_model_json = open('data/pretrained_models/text_model.json', 'r')
loaded_text_model = text_model_json.read()
text_model_json.close()
text_model = model_from_json(loaded_text_model)
text_model.load_weights('data/pretrained_models/text_model.h5')

video_model_json = open('data/pretrained_models/video_model.json', 'r')
loaded_video_model = video_model_json.read()
video_model_json.close()
video_model = model_from_json(loaded_video_model)
video_model.load_weights('data/pretrained_models/video_model.h5')

audio_model_json = open('data/pretrained_models/audio_model.json', 'r')
loaded_audio_model = audio_model_json.read()
audio_model_json.close()
audio_model = model_from_json(loaded_audio_model)
audio_model.load_weights('data/pretrained_models/audio_model.h5')

# Load the tokenizer

with open('utils/tokenizer.json') as f:
    data = json.load(f)
    tokenizer = Tokenizer()
    tokenizer.word_index = data

# Text Sentiment Prediction

with open('data/extracted_files/text.txt', 'r') as f:
    text_data = f.read()
    text_data = text_data.split('\n')

text_preds = []

for sentence in text_data:
    sentence = [sentence]
    sequence = tokenizer.texts_to_sequences(sentence)
    text = pad_sequences(sequence, maxlen=25, padding='post')

    pred = text_model.predict(text, verbose=0)
    text_preds.append(pred[0][0])

sentiment_score["text"] = np.mean(text_preds)

# Video Sentiment Prediction

video_preds = []

folder_path = 'data/extracted_files/frames'

for filename in os.listdir(folder_path):
    filepath = os.path.join(folder_path, filename)
    frame = cv2.imread(filepath)
    face_detector = cv2.CascadeClassifier('utils/haarcascade_frontalface_default.xml')
    num_faces = face_detector.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=3)
    
    for (x, y, w, h) in num_faces:
        roi_gray_frame = frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        vid_pred = video_model.predict(cropped_img, verbose=0)
        maxindex = np.argmax(vid_pred)
        video_preds.append(vid_pred[0][maxindex])

sentiment_score["video"] = np.mean(video_preds)

# Audio Sentiment Prediction

audio_data = preprocess_audiofile('data/extracted_files/audio.wav')

audio_segments = []

for i in range(0, len(audio_data), 2145):
    segment = audio_data[i:i+2145]
    if len(segment) < 2145:
        padding = 2145 - len(segment)
        segment = np.pad(segment, (0, padding), mode='constant')
    audio_segments.append(segment)

reshaped_audio_segments = []

for seg in audio_segments:
    reshaped_audio_segments.append(seg.reshape((165, 13)))

reshaped_audio_segments = np.array(reshaped_audio_segments)

audio_preds = []

audio_pred = audio_model.predict(reshaped_audio_segments, verbose=0)
for i in range(len(audio_pred)):
    maxindex = np.argmax(audio_pred[i])
    audio_preds.append(audio_pred[i][maxindex])

sentiment_score["audio"] = np.mean(audio_preds)

# Compound score
weights = [0.1, 0.5, 0.4]
compound_score = np.dot(weights, list(sentiment_score.values()))
sentiment_score["compound"] = compound_score

print("\n",sentiment_score)

if sentiment_score['compound'] > 0.55:
    print("The sentiment of the video is Positive")
elif sentiment_score['compound'] < 0.45:
    print("The sentiment of the video is Negative")
else:
    print("The sentiment of the video is Neutral")