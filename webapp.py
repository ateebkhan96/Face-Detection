import streamlit as st
import torch
import numpy as np
import cv2

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp/weights/last.pt', force_reload=False)

def image_detection():
    # File uploader to upload an image
    uploaded_file = st.file_uploader("Upload Image :")

    if uploaded_file is not None:
        # Read and process the uploaded image
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run the YOLOv5 model on the image
        results = model(img)
        output = np.squeeze(results.render())
        
        # Display the output image with detected faces
        st.image(output, caption='Output Image', use_column_width=True)
    else:
        st.warning("Please upload an image!")

    st.write("Thank you for using Face Detection Model")


def main():
    # Header for the app
    st.header("Face Detection WebApp by Ateeb Khan")
    # Run the image detection function
    image_detection()


if __name__ == "__main__":
    main()
