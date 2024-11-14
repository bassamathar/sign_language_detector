from ultralytics import YOLO
import cv2
import numpy as np

detected_class_names = []
def gen_camera(detected_class_names):

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened(): 
        print("Error: Could not access the camera.")
        return
    
    class_names = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", 
        "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
    ]

    model = YOLO('bestllee.pt') 
    
    while True:
        ret, frame = cap.read()  
        if not ret:
            print("Failed to grab frame, exiting...")
            break
        
        result = model(frame, stream=True)  

        for r in result:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 0, 50), 3)
                class_idx = int(box.cls)
                class_name = class_names[class_idx]
                cv2.putText(frame, class_names[class_idx], (x1 + 10, y1), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 200, 200), 2)
                detected_class_names.append(class_name)
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        

    cap.release()  


def values(detected_class_names):
    print(detected_class_names)
          
