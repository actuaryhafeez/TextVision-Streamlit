import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image

# Initialize the EasyOCR Reader
reader = easyocr.Reader(['en'])

def draw_results(image_path, results):
    img = cv2.imread(image_path)

    # Create an empty list to store result texts for later display
    result_texts = []

    for detection in results:
        coordinates = detection[0]
        top_left = tuple(map(int, coordinates[0]))
        bottom_right = tuple(map(int, coordinates[2]))
        text = detection[1]
        confidence = detection[2]

        # Draw rectangle around detected text
        img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)
        
        # Append the detected text, its confidence, and coordinates to our result list
        result_texts.append((text, coordinates, confidence))
    
    return img, result_texts

st.title("TextVision Streamlit App")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    # st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button('Process Image'):
        # Save the uploaded image to a temporary file
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_image.getbuffer())

        # Process the image with EasyOCR
        results = reader.readtext("temp_image.jpg")

        # Get processed image and result texts
        processed_image, result_texts = draw_results("temp_image.jpg", results)

        # Organize the layout in two columns
        col1, col2 = st.columns(2)
                
        # Display original image and processed image side by side
        with col1:
            st.image(uploaded_image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(processed_image, caption="Processed Image", use_column_width=True, channels="BGR")
                
        # Display results below the images
        st.write("### OCR Results:")
        for text, coordinates, confidence in result_texts:
            st.write(f"Text: {text}")
            st.write(f"Coordinates: {coordinates}")
            st.write(f"Confidence: {confidence * 100:.2f}%")
    st.write("--------------")

st.write("Upload an image to get started!")
