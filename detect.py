import numpy as np
import pandas as pd
import cv2
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report

# Function to load and preprocess images
def get_and_preprocess(path):
    normal = glob.glob(f'{path}/NORMAL/*')
    pneumonia = glob.glob(f'{path}/PNEUMONIA/*')
    X = []
    y = []
    for i in normal:
        image = cv2.imread(i, 0)
        image = cv2.resize(image, (128, 128))
        image = image / 255
        X.append(image)
        y.append(0)  # 0 for normal
    for i in pneumonia:
        image = cv2.imread(i, 0)
        image = cv2.resize(image, (128, 128))
        image = image / 255
        X.append(image)
        y.append(1)  # 1 for pneumonia
    return np.array(X), np.array(y)

# Load and preprocess data
X_train, y_train = get_and_preprocess('C:\\Users\\Ashish Sugunan\\OneDrive\\Desktop\\mini-project\\archive (1)\\chest_xray\\train')
X_val, y_val = get_and_preprocess('C:\\Users\\Ashish Sugunan\\OneDrive\\Desktop\\mini-project\\archive (1)\\chest_xray\\val')
X_test, y_test = get_and_preprocess('C:\\Users\\Ashish Sugunan\\OneDrive\\Desktop\\mini-project\\archive (1)\\chest_xray\test')

# Expand dimensions for CNN input
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=2)
y_val = to_categorical(y_val, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPool2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPool2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation
train_datagen = ImageDataGenerator(rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Fit model
history = model.fit(train_datagen.flow(X_train, y_train, batch_size=32),
                    steps_per_epoch=len(X_train) / 32,
                    epochs=50,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping])

# Evaluate model on test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# Predictions
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion matrix and classification report
conf_matrix = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_report(y_true, y_pred))
