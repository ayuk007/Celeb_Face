import pickle
import streamlit as st
import os
import cv2
import numpy as np

from PIL import Image
from mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.all_utils import read_yaml, create_directory

st.set_page_config(
    page_title= "Celeb Face_Match"
)

import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('D:\Projects\Face_Match\pexels-johannes-plenio-1103970.jpg') 

st.title('_:blue[Celeb Match!!]_ :sunglasses:')

st.markdown(
    """Welcome to the Celebrity Face Match project, a cutting-edge technology that allows you to find your celebrity lookalike within seconds! 
    Have you ever wondered which famous actor or actress resembles you the most? With our state-of-the-art algorithm, you can upload a photo of yourself and instantly see which celebrity shares your facial features.

This project utilizes the latest advances in artificial intelligence and machine learning to accurately match your facial characteristics with those of hundreads of celebrities. Our database is constantly updated, ensuring that you have access to the most current and relevant results.

Not only is this project fun and entertaining, but it also serves as a testament to the incredible progress we have made in the field of computer vision. So, whether you're curious about which Bollywood or Hollywood A-lister you most closely resemble or simply looking for a new source of entertainment, the Celebrity Face Match project is sure to impress.
    """
)