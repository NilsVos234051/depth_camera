
import pyrealsense2 as rs
import numpy as np
import cv2
from hoodie_overlay import overlay_hoodie

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Load person detector
net = cv2.dnn.readNetFromCaffe(
    cv2.data.haarcascades + "../../demo_mode/MobileNetSSD_deploy.prototxt",
    cv2.data.haarcascades + "../../demo_mode/MobileNetSSD_deploy.caffemodel"
)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        (h, w) = color_image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(color_image, (300, 300)),
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

                # Use center point for depth measurement
                center_x = (startX + endX) // 2
                center_y = (startY + endY) // 2
                distance = depth_image[center_y, center_x] * 0.001  # convert to meters

                show_hoodie = distance < 1.5
                color_image = overlay_hoodie(color_image, (startX, startY, endX, endY), show=show_hoodie)

                cv2.rectangle(color_image, (startX, startY), (endX, endY), (255, 0, 0), 2)
                cv2.putText(color_image, f"Distance: {distance:.2f}m", (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("RealSense Hoodie Overlay", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
