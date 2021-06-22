import streamlit as st
import keras
import numpy as np
import cv2
from PIL import Image

model = keras.models.load_model('Datasets/best_model.hdf5')

st.title("Covid-19 Detection using chest X-ray classification")

file = st.file_uploader(label='Upload the image containing chest X-ray', type=['jpg','jpeg','bmp'])

def import_and_predict(image, model):
    img = np.asarray(image)
    img = cv2.resize(img,(224,224))
    img = img/255.0
    img = np.expand_dims(img,axis=0)
    prediction = model.predict(img)
        
    return prediction
    
    
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    if prediction <= 0.5:
        st.write("Covid-19 Positive")
    else:
        st.write("Covid-19 Negative")
    
    st.text("Probability (0: Covid-19 Positive, 1: Covid-19 Negative)")
    st.write(prediction)

