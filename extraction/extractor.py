import os
from dotenv import load_dotenv, find_dotenv
from moviepy.editor import *
import pytesseract
from PIL import Image

load_dotenv(find_dotenv())

uploaded_file_path = os.environ.get("UPLOADED_FILE_PATH")
extracted_file_path = os.environ.get("EXTRACTED_FILE_PATH")
tesseract_file_path = os.environ.get("TESSERACT_FILE_PATH")

pytesseract.pytesseract.tesseract_cmd = tesseract_file_path

def extract_audio(
        video_path=f"{uploaded_file_path}/video.mp4",
        audio_path=f"{extracted_file_path}/audio.wav"
):
    
    print("Extracting Audio...")

    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path, logger=None)

    print("Audio Extracted!!")

def extract_frames(
        video_path=f"{uploaded_file_path}/video.mp4",
        frames_path=f"{extracted_file_path}/frames"
):
    
    print("\nExtracting Videoframes...")

    os.makedirs(frames_path, exist_ok=True)
    video = VideoFileClip(video_path)
    frames = video.write_images_sequence(os.path.join(frames_path, 'frame%02d.png'), fps=2, logger=None)

    print("Videoframes Extracted!!")

def extract_text(
        text_path=f"{extracted_file_path}/text.txt"
):
    
    print("\nExtracting Text...")

    frames_path=f"{extracted_file_path}/frames"
    text = ""

    for filename in os.listdir(frames_path):
        img_path = os.path.join(frames_path, filename)
        img = Image.open(img_path)
        text += pytesseract.image_to_string(img)

    with open(text_path, 'w') as file:
        file.write(text)

    print("Text Extracted!!")

def extract(
        video_name: str
):
    
    video_path=f"{uploaded_file_path}/{video_name}"    
    os.makedirs(extracted_file_path, exist_ok=True)

    print("Extracting...\n")
    extract_audio(video_path)
    extract_frames(video_path)
    extract_text()
    print(f"\nExtraction Completed. Files stored in \'{extracted_file_path}/\'")