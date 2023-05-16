from ultralytics import YOLO
import cv2

model = YOLO("Yolo_weights/yolov8l.pt")
results = model("images/im2.jpg", show=True)
cv2.waitKey(0)
