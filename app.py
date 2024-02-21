 ## For huggingsface models --- "https://huggingface.co/tasks"


from dotenv import load_dotenv
from transformers import pipeline
import google.generativeai as genai
from PIL import Image
import os
import requests
import streamlit as st

load_dotenv()

os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

# Set the path to the directory containing pre-downloaded models
PRETRAINED_MODEL_DIR = '/path/to/pretrained_models'

# Image To Text Model
def image2text(url):
    image_to_text = pipeline('image-to-text', model=os.path.join(PRETRAINED_MODEL_DIR, 'Salesforce/blip2-opt-2.7b'), use_fast=True)

    text = image_to_text(url)[0]['generated_text']

    print(text)
    return text

# LLM
template = '''
             You are a story teller:
             You can generate a short story based on a simple narrative, the story should be not more than 30 words.
'''

def get_gemini_response(input, prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([input, prompt])
    return response.text

# Text to Speech
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

    payloads = {
         'inputs': message
    }

    response = requests.post(API_URL, json=payloads, headers=headers)
        
    with open('audio.flac', 'wb') as file:
        file.write(response.content)


input_text = "Sample input text"
story = get_gemini_response(input_text, template)
text2speech(story)




input = image2text('image.jpeg')
story = get_gemini_response(input,template)
text2speech(story)




 


 
