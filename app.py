import streamlit as st
import pickle
from PIL import Image
import numpy as np
import cv2
from mtcnn import MTCNN

@st.cache_resource
def load_face_mask_model(model_path):
    """Load the pre-trained face mask detection model."""
    with open(model_path, 'rb') as file:
        return pickle.load(file)

# Load the model
model = load_face_mask_model("model2.pkl") 
LABELS = ["No Mask", "Mask"] 
detector = MTCNN()  # Initialize the MTCNN face detector

def detect_mask(image):
    """Detect faces in the image and predict mask status."""
    # Convert the image to a numpy array
    image_array = np.array(image)

    # Detect faces in the image using MTCNN
    results = detector.detect_faces(image_array)
    if len(results) == 0:
        return "No face detected", None, None

    # Extract the first detected face's bounding box
    x, y, width, height = results[0]['box']
    x, y = max(0, x), max(0, y)  # Ensure the box is within image bounds
    face = image_array[y:y+height, x:x+width]

    # Convert face to grayscale
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Resize dynamically based on the model's expected input shape
    input_shape = model.input_shape  
    target_size = input_shape[1:3] 
    resized_face = cv2.resize(face_gray, target_size)

    # Normalize and reshape for model input
    input_array = resized_face.reshape(1, target_size[0], target_size[1], 1)  # Add batch and channel dimensions
    input_array = input_array / 255.0  # Normalize pixel values

    # Predict mask status using the model
    prediction = model.predict(input_array)[0]
    label = LABELS[np.argmax(prediction)]
    confidence = np.max(prediction)
    return label, confidence, face

# Streamlit interface
st.title("Face Mask Detection App")
st.write("Upload an image to detect whether the person is wearing a mask.")

# File uploader for user to upload images
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("Processing...")

    # Run the mask detection function
    label, confidence, face = detect_mask(image)

    if label == "No face detected":
        st.error(label)
    else:
        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {confidence:.2f}")

        # Display the cropped face
        if face is not None:
            st.image(face, caption="Detected Face", use_container_width=False)
