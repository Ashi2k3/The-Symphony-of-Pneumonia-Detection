from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Initialize Flask application
app = Flask(__name__, static_folder='static')


# Set up upload folder
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the trained model
model = load_model('pneumonia.h5')

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path, 0)
    img = cv2.resize(img, (128, 128))
    img = img / 255
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

# Function to classify the image
def classify_image(image_path):
    img = preprocess_image(image_path)
    pred = np.argmax(model.predict(img), axis=1)
    return "NORMAL" if pred == 0 else "PNEUMONIA"

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)
    prediction = classify_image(image_path)
    return jsonify({'prediction': prediction})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
