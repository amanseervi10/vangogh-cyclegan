import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
import base64

st.title("Image Inference with Streamlit and Flask")

# File upload widget
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Resize the image to 512x512 pixels
    image = image.resize((512, 512))

    st.image(image, caption="Resized Image", use_column_width=True)

    if st.button("Vangohify"):
        # Convert the resized image to a numpy array
        input_data = np.array(image, dtype=np.float32)
        input_data = (input_data / 255.0 - 0.5) / 0.5  # Normalize
        input_data = input_data.transpose(2, 0, 1)  # Transpose to (channels, height, width)
        input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

        # Send the image data to the Flask backend for processing
        response = requests.post("http://localhost:5000/vangoghify", json={"input_data": input_data.tolist()})

        if response.status_code == 200:
            # print(response.json())
            result_image_base64 = response.json()['output_image']
            result_image_bytes = base64.b64decode(result_image_base64)

            # Convert the binary data to a NumPy array
            result_image_data = np.frombuffer(result_image_bytes, dtype=np.uint8)
            result_image = np.reshape(result_image_data, (512, 512, 3))

            # Convert the NumPy array to a PIL image
            result_image = Image.fromarray(result_image)

            # Display the processed image
            st.image(result_image, caption="Transformed Image", use_column_width=True)
        else:
            st.error("Transformation failed.")