import streamlit as st
import torch
import numpy as np
import cv2

# Directly load the model using torch.load
@st.cache_resource  # Caching the model to avoid reloading it repeatedly
def load_model():
    model = torch.load('yolov5/runs/train/exp/weights/last.pt', map_location=torch.device('cpu'))  # Use CPU for Streamlit
    model.eval()  # Set model to evaluation mode
    return model

model = load_model()

def image_detection():
    uploaded_file = st.file_uploader("Upload Image :")

    if uploaded_file is not None:
        # Read and process the uploaded image
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Prepare image for the model
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # Convert to tensor and normalize
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        # Run the model on the image
        with torch.no_grad():
            results = model(img_tensor)[0]
        
        # Convert the model output back to image format and display
        output = img  # Use results from the model as needed for post-processing
        st.image(output, caption='Output Image', use_column_width=True)
    else:
        st.warning("Please upload an image!")

    st.write("Thank you for using Face Detection Model")

def main():
    st.header("Face Detection WebApp by Ateeb Khan")
    image_detection()

if __name__ == "__main__":
    main()
