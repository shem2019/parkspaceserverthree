

#then import the packages to the project
from ultralytics import YOLO
import cv2
import cvzone
import math

import numpy as np
# initialization
count=0
tracking_objects={}
tracking_id=0


center_points_prev_frame=[]


"""""
for camera video
cap =cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4, 480)
"""
#for video video capture that is the video below
cap = cv2.VideoCapture("../Videos/cars.mp4")

#here we select the model we will use to classify objects in a frame of the video. A frame is an one instance of a video
model=YOLO("Yolo_weights/yolov8n.pt")
# this is just a python list of class names . note how lists are writen.
classNames= ["person", "bicycle", "car", "motorbike",
             "airplane", "bus", "train", "truck", "boat", "traffic light",
             "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
             "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
             "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
             "skis", "snowboard", "sports ball", "kite", "baseball bat",
             "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
             "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
             "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
             "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
             "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
             "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
             "teddy bear", "hair drier", "toothbrush"]
# this is a conditional statement that tells the computer to do a list of things when a condition is met.
# here it is telling it to analyse the images as long as the video is running
while True:
    success, img =cap.read()
    count+=1
    results= model(img, stream=True)

    center_points_cur_frame=[]
# this is code for drawing bounding boxes for identified objects in a frame
    for r in results:
        boxes= r.boxes
        for box in boxes:
            #bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2= int(x1),int(y1),int(x2),int(y2)
            w, h = x2 - x1, y2 - y1


            cvzone.cornerRect(img, (x1, y1, w, h))
            # confidence level ; this is how sure the model is that the object is what it classifies it as
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(0, y1 - 20)), scale=0.6,
                               thickness=1, offset=3)
            #print("FRAME N°", count, "", x1, y1, x2, y2)
            #center points
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            center_points_cur_frame.append((cx, cy))

        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance= math.hypot(pt2[0] - pt[0], pt2[1]-pt[1])

                if distance < 15:
                    tracking_objects[tracking_id]=pt
                    tracking_id+=1


        for object_id, pt in tracking_objects.items():
            cv2.circle(img,pt,5, (0,0,255), -1)
            cv2.putText(img,str(object_id), (pt[0],pt[1]-7,), 0,1 ,(0,0,255),2)



        print("tracking objects")
        print(tracking_objects)


        print(" cur frame")
        print(center_points_cur_frame)
        print("Prev frame")
        print(center_points_prev_frame)

    cv2.imshow("image",img)
    center_points_prev_frame = center_points_cur_frame.copy()
    #this code tells compliter how long to wait before going to next frame
    cv2.waitKey(0)



