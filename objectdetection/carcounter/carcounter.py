from ultralytics import YOLO
import cv2
import cvzone
import math

#for video
cap =cv2.VideoCapture("../Videos/cars.mp4")

model=YOLO("Yolo_weights/yolov8n.pt")
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
mask = cv2.imread("mask.png")
while True:
    success, img =cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results= model(imgRegion, stream=True)




    for r in results:
        boxes= r.boxes
        for box in boxes:
            #bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2= int(x1),int(y1),int(x2),int(y2)
           # print("x1,y1,x2,y2")
           # cv2.rectangle(img,(x1,y1),(x2,y2),(255,200,255),3)
            w,h= x2-x1,y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h),l=9)
            #confidence level
            conf= math.ceil((box.conf[0]*100))/100
            print(conf)
           #cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(0,y1-20)))
            #class names
            cls = int( box.cls[0])
            currentClass=classNames[cls]
            if currentClass == "car" or currentClass == "motorbike" \
                    or currentClass == "truck" or currentClass == "bus" and conf>0.3:
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                                 scale=0.6,thickness=1,offset=3)




    cv2.imshow("image",img)
    cv2.imshow("ImageRegion",imgRegion)
    cv2.waitKey(1)

