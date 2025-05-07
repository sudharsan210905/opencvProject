import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr

st.set_page_config(page_title="Number Plate Detector", layout="centered")

st.title("🔍 Number Plate Detection")
st.write("Upload an image or use webcam to detect number plates.")

reader = easyocr.Reader(['en'])

def detect_number_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    plate = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            plate = image[y:y+h, x:x+w]
            break

    if plate is not None:
        result = reader.readtext(plate)
        text = result[0][1] if result else "Text not detected"
        return plate, text
    else:
        return None, "Number plate not found"

upload_option = st.radio("Choose Input", ["Upload Image", "Use Webcam"])

if upload_option == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        plate_img, text = detect_number_plate(image)

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image")
        if plate_img is not None:
            st.image(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB), caption="Detected Number Plate")
        st.success(f"Detected Text: {text}")

else:
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            break
        plate_img, text = detect_number_plate(frame)
        if plate_img is not None:
            frame = cv2.rectangle(frame, (0,0), (300,30), (0,0,0), -1)
            cv2.putText(frame, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        cap.release()
