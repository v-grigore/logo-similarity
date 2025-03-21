import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
from tqdm import tqdm

from src.constants import *

def preprocess_image(image_path, target_size=(224, 224)):
    try:
        img = Image.open(image_path).convert("RGBA")
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

    # Get original size and calculate aspect ratio
    original_width, original_height = img.size
    aspect_ratio = original_width / original_height

    # Resize while maintaining aspect ratio
    if aspect_ratio > 1:  # Wider than tall
        new_width = target_size[0]
        new_height = int(target_size[0] / aspect_ratio)
    else:  # Taller than wide
        new_height = target_size[1]
        new_width = int(target_size[1] * aspect_ratio)

    img_resized = img.resize((new_width, new_height))

    img_arr = np.array(img_resized)
    alpha = img_arr[:, :, 3]
    
    # Convert to grayscale (RGB -> Luminance)
    grayscale = 0.299 * img_arr[:, :, 0] + 0.587 * img_arr[:, :, 1] + 0.114 * img_arr[:, :, 2]
    # Mask out transparent pixels

    grayscale = grayscale[alpha > 0]
    avg_brightness = np.mean(grayscale)

    # Choose padding: white for dark images, black for bright ones
    padding_color = (255, 255, 255) if avg_brightness < 50 else (0, 0, 0)

    # Create RGB background (no alpha channel)
    padded_img = Image.new("RGB", target_size, padding_color)
    
    # Paste resized image using alpha channel as mask
    padded_img.paste(
        img_resized, 
        ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2),
        mask=img_resized.split()[3]  # Use alpha channel as mask
    )

    img = np.array(padded_img) / 255.0
    return img

def extract_features_helper(image_path, model):
    img = preprocess_image(image_path)
    if img is None:
        return None
    img_batch = np.expand_dims(img, axis=0)
    img_batch = preprocess_input(img_batch)
    features = model(img_batch)
    # Flatten the output from the convolutional layers into a 1D vector.
    features = tf.reduce_mean(features, axis=(1, 2)).numpy().flatten()
    return features

def extract_features_from_directory(logo_dir, model, max_logos=MAX_LOGOS):
    logo_files = [os.path.join(logo_dir, f) for f in os.listdir(logo_dir) if f.endswith(".png")]
    logo_files = logo_files[:max_logos]
    
    features = []
    valid_files = []
    
    for logo in tqdm(logo_files, desc="Processing Logos"):
        feat = extract_features_helper(logo, model)
        if feat is not None:
            features.append(feat)
            valid_files.append(logo)
    
    return np.array(features), valid_files

def save_features_and_labels(features, logo_files, filename=FEATURES_FILE):
    np.savez(filename, features=features, files=logo_files)
    print(f"Features and labels saved to {filename}.")

def load_features_and_labels(filename=FEATURES_FILE):
    data = np.load(filename)
    return data['features'], data['files']

def extract_features():
    # --- Configuration ---
    # Directory where logos are stored (update accordingly)
    logo_dir = LOGOS_DIR

    # --- Load MobileNetV2 feature extractor from Keras ---
    # Using MobileNetV2 Feature Vector (input: 224x224x3)
    model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False

    # --- Extract features using ResNet50 ---
    features, logo_files = extract_features_from_directory(logo_dir, model)

    save_features_and_labels(features, logo_files)
