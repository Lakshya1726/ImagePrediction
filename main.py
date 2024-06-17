import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image


model = ResNet50(weights='imagenet')


def predict(img):

    if img.mode != 'RGB':
        img = img.convert('RGB')


    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)


    preds = model.predict(x)
    return decode_predictions(preds, top=3)[0]



st.title('Image Classification with ResNet50')
st.write('Upload an image')


uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:

    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write('')
    st.write('Classifying...  taking time')


    predictions = predict(img)
    for pred in predictions:
        st.write(f'{pred[1]}: {pred[2] * 100:.2f}%')


