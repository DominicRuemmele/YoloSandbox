# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:02:07 2024

@author: domin
"""

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2

model = YOLO('yolov8n.pt') # load model
img = cv2.imread(r'start.jpg') # prefix path with r (if using full path this is required for '/*')

results = model.predict(img)

for r in results:
    illustrator = Annotator(img)
    boxes = r.boxes
    for box in boxes:
        if(box.conf[0] < 0.5): # only caption objects with > 50% confidence
            continue
        else:
            bounds = box.xyxy[0]  # bounding boxes from model predictions
            nameIndex = box.cls
            if(model.names[int(nameIndex)] == "person"):
                illustrator.box_label(bounds, "Parrot Head",color=(255, 0, 0), txt_color=(255, 255, 255)) # note colors are not (R,G,B) but instead (B,G,R)
            else:
                illustrator.box_label(bounds, model.names[int(nameIndex)],color=(0, 255, 0), txt_color=(255, 255, 255))

img = illustrator.result()  
cv2.imwrite(r'image.png', img)

