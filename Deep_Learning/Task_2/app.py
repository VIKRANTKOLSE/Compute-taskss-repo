import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


CLASS_NAMES = [
    'burger', 'butter_naan', 'chai', 'chapati', 'chole_bhature',
    'dal_makhani', 'dhokla', 'fried_rice', 'idli', 'jalebi',
    'kaathi_rolls', 'kadai_paneer', 'kulfi', 'masala_dosa', 'momos',
    'paani_puri', 'pakode', 'pav_bhaji', 'pizza', 'samosa'
]


@st.cache_resource
def load_food_model():
    # 1. Rebuild the exact data augmentation block used in training
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ])
    
    # 2. Rebuild the ResNet50V2 base model
    # Note: weights=None because we don't need to download ImageNet weights again
    conv_base = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3)
    )
    
    # 3. Assemble the full Sequential model exactly as done in the notebook
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(224, 224, 3)),
        data_augmentation,
        tf.keras.layers.Rescaling(1./255),
        conv_base,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(265, activation='relu'),
        tf.keras.layers.Dense(20, activation='softmax')
    ])
    
    # 4. Load ONLY the weights from the .keras file to bypass the architecture loading bug
    model.load_weights('resnet_model.keras')
    
    return model

st.set_page_config(page_title="Indian Food Classifier", page_icon="🍛")

st.title("Indian Food Image Classifier 🍛")
st.write("Upload an image, and the model will predict which of the 20 food categories it belongs to.")

# Load the model
try:
    model = load_food_model()
except Exception as e:
    st.error(f"Error loading model. Ensure 'resnet_model.keras' is in the same directory. Details: {e}")
    st.stop()

# 3. File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB") # Convert to RGB to avoid alpha channel issues
    st.image(image, caption='Uploaded Image', use_container_width=400)

    st.write("Classified")

    # 4. Preprocess the image
    # Resize to match the 224x224 input shape expected by ResNet50V2
    image_resized = image.resize((224, 224))
    
    # Convert the PIL image to a NumPy array
    image_array = tf.keras.preprocessing.image.img_to_array(image_resized)
    
    # Expand dimensions to create a batch of 1: shape becomes (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)
    
    # Note: Your model architecture already includes `tf.keras.layers.Rescaling(1./255)`, 
    # so we pass the raw pixel values (0-255) directly to the predict function.

    # 5. Make the prediction
    predictions = model.predict(image_array)
    
    # Extract the highest probability and the corresponding class index
    predicted_class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    predicted_class_name = CLASS_NAMES[predicted_class_idx]

    # 6. Display results
    st.success(f"**Prediction:** {predicted_class_name.replace('_', ' ').title()}")
    st.info(f"**Confidence:** {confidence:.2f}%")