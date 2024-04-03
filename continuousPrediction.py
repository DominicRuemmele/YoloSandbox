from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
model = YOLO('yolov8n.pt')

#%%

cap = cv2.VideoCapture(0)

#set frame width and height
cap.set(3, 640)
cap.set(4, 480)

while True:
    _, img = cap.read()
    
    results = model.predict(img)

    for r in results:
        illustrator = Annotator(img)
        boxes = r.boxes
        for box in boxes:
            if(box.conf[0] < 0.5):
                continue
            else:
                bounds = box.xyxy[0]  # bounding boxes from model predictions
                nameIndex = box.cls
                if(model.names[int(nameIndex)] == "person"):
                    illustrator.box_label(bounds, "Parrot Head",color=(255, 0, 0), txt_color=(255, 255, 255))
                else:
                    illustrator.box_label(bounds, model.names[int(nameIndex)],color=(0, 255, 0), txt_color=(255, 255, 255))
          
    img = illustrator.result()  
    cv2.imshow('frame', img)     
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#cv2.imwrite(r'C:\Users\domin\Downloads\image.png', img)
cap.release()
cv2.destroyAllWindows()