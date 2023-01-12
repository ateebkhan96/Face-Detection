import streamlit as st
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer


model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp/weights/last.pt', force_reload=True)


def webcam_detection():
    start_button = st.button('Start webcam detection')
    stop_button = st.button('Stop webcam detection')
    FRAME_WINDOW = st.image([])
    #camera = cv2.VideoCapture(0)

    if start_button:
        run = True
        while run:
            # _, frame = camera.read()
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # results = model(frame)
            # output = np.squeeze(results.render())
            frame = webrtc_streamer(key="face")
            frame = np.array(frame)
            results = model(frame)
            output = np.squeeze(results.render()) 
            FRAME_WINDOW.image(output)
            if stop_button:
                run = False
                st.write("Webcam has stopped")
    else:
        st.warning("Press start button to start webcam detection")


def image_detection():
    

    uploaded_file = st.file_uploader("Upload Image :")

    if uploaded_file is not None:
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img)
        output = np.squeeze(results.render())
        st.image(output, caption='Output Image', use_column_width=True)
    else:
        st.warning("Please upload an image!")

    st.write("Thank you for using Face Detection Model")

def main():
    st.header("Face Detection WebApp by Ateeb Khan")
    st.subheader("Select the option below to detect")
    option = st.selectbox("Select one option",["Image", "Webcam"])
    if option == "Image":
        image_detection()
    else:
        webcam_detection()

if __name__ == "__main__":
    main()
