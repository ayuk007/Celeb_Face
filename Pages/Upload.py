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
from collage import create_collage
from sketch import create_sketch

st.set_page_config(
    page_title = "Celeb Face_Match"
)

config = read_yaml('config/config.yaml')
params = read_yaml('params.yaml')

artifacts = config['artifacts']
artifacts_dir = artifacts['artifacts_dir']

#upload
upload_image_dir = artifacts['upload_image_dir']
upload_path = os.path.join(artifacts_dir, upload_image_dir)

#pickle format data dir
pickle_format_data_dir = artifacts['pickle_format_data_dir']
image_pickle_file_name = artifacts['image_pickle_file_name']

raw_local_dir_path = os.path.join(artifacts_dir, pickle_format_data_dir)
pickle_file = os.path.join(raw_local_dir_path, image_pickle_file_name)

#feature path
feature_extractor_dir = artifacts['feature_extraction_dir']
extracted_features_name = artifacts['extracted_features_name']

feature_extraction_path = os.path.join(artifacts_dir, feature_extractor_dir)
feature_name = os.path.join(feature_extraction_path, extracted_features_name)

model_name = params['base']['BASE_MODEL']
include_tops = params['base']['include_top']
poolings = params['base']['pooling']

detector = MTCNN()
model = VGGFace(model = model_name, include_top = include_tops, input_shape = (224,224,3), pooling = poolings)
filenames = pickle.load(open(pickle_file, 'rb'))
feature_list = pickle.load(open(feature_name, 'rb'))



#save upload image
def save_upload_image(uploadImage):
    try:
        create_directory(dirs = [upload_path])

        with open(os.path.join(upload_path, uploadImage.name), 'wb') as f:
            f.write(uploadImage.getbuffer())
        return True
    except:
        return False

#Extracted Feature
def extract_feature(img_path, model, detector):
    img = cv2.imread(img_path)
    result = detector.detect_faces(img)

    x, y, width, height = result[0]['box']
    face = img[y:y + height, x:x + width]

    #extract features
    image = Image.fromarray(face)
    image = image.resize((224,224))

    face_array = np.asarray(image)
    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis = 0)
    preprocess_img = preprocess_input(expanded_img)
    result = model.predict(preprocess_img).flatten()

    return result

def recommend(feature_list, features):
    similarity = []

    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])
    
    index_pos = sorted(list(enumerate(similarity)), reverse = True, key = lambda x: x[1])[0][0]
    return index_pos, max(similarity)

#streamlit

st.title('_:blue[Celeb Match!!]_ :sunglasses:')

def add_bg_from_url():
    st.markdown(
         f"""
         <marquee loop = "1"><blink>Who is your matching celebrity..?</blink> </marquee>

         <style>
         .stApp {{
             background-image: url("https://cdn.pixabay.com/photo/2019/04/24/11/27/flowers-4151900_960_720.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()


# import base64
# def add_bg_from_local(image_file):
#     with open(image_file, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read())
#     st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
#         background-size: cover
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
#     )
# add_bg_from_local('blue_bg.png')    

uploadImage = st.file_uploader('Choose an image')

if uploadImage is not None:
    # save the image
    if save_upload_image(uploadImage):
        #load image
        display_image = Image.open(uploadImage)

        #extract feature
        features = extract_feature(os.path.join(upload_path, uploadImage.name), model, detector)

        
        #match
         
        index_pos, conf = recommend(feature_list, features)
        

        predictor_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))

        #display_image
        col1, col2 = st.columns(2)

        st.header(f"Your face is matched with {predictor_actor} by {round(conf * 100, 2)}% confidence")

        img = create_collage(cv2.cvtColor(np.asarray(display_image), cv2.COLOR_BGR2RGB), cv2.imread(filenames[index_pos]))
        st.image(img)
        count = len(os.listdir("Uploaded_Images/Colored"))
        cv2.imwrite(f"Uploaded_Images/Colored/{count+1}.jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if st.button('Create Sketch'):
            sk = create_sketch(img)
            st.image(sk)
            name = st.text_input("Enter your name")
            if name!=None:
                count = len(os.listdir("Uploaded_Images/Sketch"))
                cv2.imwrite(f"Uploaded_Images/Sketch/{count+1}.jpg", sk)

        # with col1:
        #     st.header('Your uploaded image')
        #     st.image(display_image.resize((350,300)))

        # with col2:
        #     st.header('Seems like ' + predictor_actor)
        #     actor = Image.open(filenames[index_pos]).resize((350,300))
        #     st.image(actor)