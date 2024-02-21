## For huggingsface models --- "https://huggingface.co/tasks"


from dotenv import load_dotenv
from transformers import pipeline
import google.generativeai as genai
from PIL import Image
import os
import requests
import streamlit as st
from io import BytesIO

load_dotenv()

os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

## Image To Text Model
def image2text(url):
    image_to_text = pipeline('image-to-text', model='Salesforce/blip2-opt-2.7b', use_fast=True)

    text = image_to_text(url)[0]['generated_text']

    print(text)
    return text

## LLM
template = '''
             You are a story teller:
             You can generate a short story based on a simple narrative, the story should be not more than 30 words.
'''

def get_gemini_response(input, prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([input, prompt])
    return response.text

## Text to Speech
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

    payloads = {
         'inputs':message
    }

    response = requests.post(API_URL, json=payloads, headers=headers)
        
    with open('audio.flac', 'wb') as file:
        file.write(response.content)


def main():
    st.set_page_config(page_title='Image To Audio Story')
    st.header('Turn Image into Audio Story')

    upload_file = st.file_uploader('Choose an Image', type=["png"])

    if upload_file is not None:
        print(upload_file)
        bytes_data = upload_file.getvalue()

        with open(upload_file.name, 'wb') as file:
            file.write(bytes_data)

        st.image(upload_file, caption='Uploaded Image', use_column_width=True)

        input = image2text(upload_file.name)
        story = get_gemini_response(input, template)
        text2speech(story)

        with st.expander('Scenario'):
            st.write(input)
        with st.expander('story'):
            st.write(story)
        st.audio('audio.flac')


if __name__ == '__main__':
    main()
