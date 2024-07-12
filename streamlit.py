import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your CNN model
model = tf.keras.models.load_model('your_model_path.h5')

# Define labels
labels = ['Normal', 'Pneumonia']

# Function to preprocess image
def preprocess_image(image):
    img = image.resize((150, 150))  # Resize image to match model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make prediction
def predict(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    return prediction

# Streamlit app
def main():
    st.title('Pneumonia Detection')

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make prediction when button is clicked
        if st.button('Predict'):
            with st.spinner('Predicting...'):
                prediction = predict(image)
                predicted_label = labels[np.argmax(prediction)]
                st.success(f'Prediction: {predicted_label}')

if __name__ == '__main__':
    main()
