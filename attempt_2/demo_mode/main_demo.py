
import cv2
import numpy as np
import os
from hoodie_overlay import overlay_hoodie
from hoodie_renderer import HoodieRenderer

# Load model files
model_dir = os.path.dirname(os.path.abspath(__file__))
prototxt = os.path.join(model_dir, "MobileNetSSD_deploy.prototxt")
model = os.path.join(model_dir, "MobileNetSSD_deploy.caffemodel")

net = cv2.dnn.readNetFromCaffe(prototxt, model)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Load video
video_path = os.path.join(model_dir, "../assets/walking_demo.mp4")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Failed to open video file at: {video_path}")
    exit()

# Load hoodie renderer
hoodie_renderer = HoodieRenderer(os.path.join(model_dir, "../assets/hoodie_model_1.obj"))

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != "person":
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            bbox_height = endY - startY
            bbox_width = endX - startX

            distance = round(2.5 - bbox_height / 200, 2)
            show_hoodie = distance < 1.5

            if show_hoodie:
                hoodie_img = hoodie_renderer.render_to_image(bbox_width, int(bbox_height * 0.6))
                top_y = max(0, startY - int(hoodie_img.shape[0] * -0.4))
                roi = frame[top_y:top_y + hoodie_img.shape[0], startX:endX]
                if roi.shape[:2] == hoodie_img.shape[:2]:
                    alpha = hoodie_img[:, :, 3] / 255.0
                    for c in range(3):
                        roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * hoodie_img[:, :, c]
                    frame[top_y:top_y + hoodie_img.shape[0], startX:endX] = roi

            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
            cv2.putText(frame, f"Distance: {distance}m", (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    resized = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
    cv2.imshow("Hoodie Overlay Demo", resized)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
