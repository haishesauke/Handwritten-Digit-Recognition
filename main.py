import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

# Load the model
model = load_model('my_model.keras')

# Streamlit app
st.title('Handwritten Digit Recognition')

upload_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if upload_file is not None:
    image = Image.open(upload_file).convert('L')
    image = image.resize((28, 28))
    image = np.array(image).reshape(1, 28, 28, 1) / 255.0

    # Make prediction
    prediction = model.predict(image)
    digit = np.argmax(prediction)
    st.write(f'Predicted Digit: {digit}')