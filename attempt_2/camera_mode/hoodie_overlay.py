
import cv2

def overlay_hoodie(frame, box, show=True):
    if not show:
        return frame
    (startX, startY, endX, endY) = box
    hoodie_height = int((endY - startY) * 0.5)
    hoodie_top = startY - hoodie_height
    hoodie_top = max(hoodie_top, 0)
    cv2.rectangle(frame, (startX, hoodie_top), (endX, startY), (0, 255, 0), -1)
    cv2.putText(frame, "Hoodie ON", (startX, hoodie_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    return frame
