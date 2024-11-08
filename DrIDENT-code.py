# Import necessary libraries
import numpy as np
import pandas as pd
import cv2
import os
import math
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle
import tf2onnx  
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load metadata and image paths
metadata = pd.read_csv('table.csv', na_values=[''])
new_data = pd.read_csv('Training_set.csv')
new_data.rename(columns={'label': 'name'}, inplace=True)
metadata.rename(columns={'rxnavImageFileName': 'filename'}, inplace=True)
combined_data = pd.concat([metadata, new_data], ignore_index=True)

# Define image directories
image_dir_1 = "rximage/image/images/gallery/original"  
image_dir_2 = "archive (2)/train" 
combined_data['image_path'] = combined_data.apply(
    lambda row: os.path.join(image_dir_1, row['filename']) if row['filename'] in metadata['filename'].values 
                else os.path.join(image_dir_2, row['filename']),
    axis=1
)

# Sample a subset of the data
combined_data_sample = combined_data.sample(frac=0.5, random_state=42).reset_index(drop=True)

# Load and preprocess images
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.resize(image, target_size)
        image = image / 255.0  # Normalize to [0, 1]
    else:
        image = np.zeros((*target_size, 3))  # Placeholder for missing images
    return image

images = np.array([load_and_preprocess_image(img_path, target_size=(224, 224)) for img_path in combined_data_sample['image_path']])
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(combined_data_sample['name'])

# Filter out placeholder images
filtered_images = np.array([img for img in images if not np.array_equal(img, np.zeros((224, 224, 3)))])
filtered_labels = np.array([labels[i] for i in range(len(labels)) if not np.array_equal(images[i], np.zeros((224, 224, 3)))])

# Filter classes with at least 20 samples
class_counts = pd.Series(filtered_labels).value_counts()
valid_classes = class_counts[class_counts >= 20].index
filtered_data = [(img, label) for img, label in zip(filtered_images, filtered_labels) if label in valid_classes]
filtered_images, filtered_labels = zip(*filtered_data)
filtered_images, filtered_labels = np.array(filtered_images), np.array(filtered_labels)

# Re-encode labels to have a contiguous range
label_mapping = {label: idx for idx, label in enumerate(valid_classes)}
filtered_labels = np.array([label_mapping[label] for label in filtered_labels])

# Create a dictionary that maps encoded labels to original names
label_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))

# Save this mapping to a file for later use
with open("label_mapping.pkl", "wb") as f:
    pickle.dump(label_mapping, f)

# Setup MobileNet model with transfer learning
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='mish')(x)
x = Dropout(0.3)(x)
predictions = Dense(len(valid_classes), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers initially
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
initial_learning_rate = 1e-4
model.compile(optimizer=Adam(learning_rate=initial_learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.4,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True
)

# Train-validation split
train_images, val_images, train_labels, val_labels = train_test_split(filtered_images, filtered_labels, test_size=0.2, random_state=42)

# Convert data to tf.data.Dataset format
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

# Class weighting
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# Cosine Annealing with Gradual Warm Restarts
def cosine_annealing_with_warm_restarts(epoch, initial_lr=1e-4, min_lr=1e-6, base_interval=10):
    current_interval = base_interval * (epoch // base_interval + 1)
    lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * (epoch % current_interval) / current_interval))
    return lr

# Callbacks
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
    EarlyStopping(monitor='val_loss', patience=12, verbose=1, restore_best_weights=True),
    ModelCheckpoint('best_model_weights.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    LearningRateScheduler(lambda epoch: cosine_annealing_with_warm_restarts(epoch, initial_lr=1e-4))
]

# Train the model
history = model.fit(
    train_dataset,
    epochs=150,
    validation_data=val_dataset,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# Fine-tuning with gradual unfreezing
unfreeze_steps = [30, 60]
for step in unfreeze_steps:
    for layer in base_model.layers[-step:]:
        layer.trainable = True
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history_fine = model.fit(
        train_dataset,
        epochs=100,
        validation_data=val_dataset,
        class_weight=class_weights_dict,
        callbacks=callbacks
    )

# Load the best weights saved during training
model.load_weights('best_model_weights.keras')

# Final evaluation
val_loss, val_accuracy = model.evaluate(val_dataset)
print(f"Final validation accuracy after fine-tuning: {val_accuracy * 100:.2f}%")

# Save the model
pickle.dump(model, open("drident_finetune.pkl", "wb"))

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('pill_recognition_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Convert to ONNX
onnx_model_path = "pill_recognition_model.onnx"
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=11)
with open(onnx_model_path, "wb") as f:
    f.write(model_proto.SerializeToString())

# Save label mapping
label_mapping_dict = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
with open("label_mapping.pkl", "wb") as f:
    pickle.dump(label_mapping_dict, f)

# Load the best weights saved during training
model.load_weights('best_model_weights.keras')

# Save the model in HDF5 format
model.save("final_pill_recognition_model.h5")

# Final evaluation (optional)
val_loss, val_accuracy = model.evaluate(val_dataset)
print(f"Final validation accuracy after fine-tuning: {val_accuracy * 100:.2f}%")

from tensorflow.keras.models import load_model

model = load_model("final_pill_recognition_model.h5")

import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image

# Load the model only once using st.cache_resource
@st.cache_resource
def load_model_once():
    return load_model("final_pill_recognition_model.h5")

model = load_model_once()

# Load label mapping
with open("label_mapping.pkl", "rb") as f:
    label_mapping = pickle.load(f)

# Preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image) / 255.0   # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Prediction function
def predict_image(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_label = label_mapping.get(predicted_class_idx, "Unknown Class")
    probabilities = predictions[0]
    return predicted_label, probabilities

# Streamlit App Layout
st.title("Pill Recognition App")
st.write("Upload an image of a pill to identify its class.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process and classify the image
    processed_image = preprocess_image(image)

    # Display processed image shape
    st.write("Processed Image Shape:", processed_image.shape)

    # Display the preprocessed image without batch dimension for visualization
    st.image(processed_image[0], caption="Preprocessed Image", use_column_width=True)

    # Get predictions
    predicted_label, probabilities = predict_image(image)

    # Display predicted class and probabilities
    st.write(f"**Predicted Class**: {predicted_label}")
    st.write("### Prediction Probabilities")
    for idx, prob in enumerate(probabilities):
        class_name = label_mapping.get(idx, "Unknown Class")
        st.write(f"{class_name}: {prob:.2f}")