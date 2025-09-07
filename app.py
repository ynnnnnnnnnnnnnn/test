import streamlit as st
import torch
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Load YOLO model (YOLOv8 pretrained on COCO dataset)
model = YOLO("yolov8n.pt")  # small model, faster for testing

st.title("🚗 Car Detection App")
st.write("Upload an image and I will detect and count cars.")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert image for YOLO (numpy format)
    img_array = np.array(img)

    # Run YOLO detection
    results = model(img_array)

    # Filter for cars (class id 2 in COCO dataset)
    car_count = 0
    annotated_frame = img_array.copy()
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 2:  # COCO class 2 = car
                car_count += 1
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, "Car", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    st.subheader(f"Detected Cars: {car_count}")

    # Show annotated image
    st.image(annotated_frame, caption="Detected Cars", use_column_width=True)
