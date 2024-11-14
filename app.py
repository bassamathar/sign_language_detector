from flask import Flask, render_template, Response
import cv2
from main1 import gen_camera, values

app=Flask(__name__)


def gen_cam():
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame, exiting...")
            break
        
        # Encode frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Convert the frame to bytes
        frame_bytes = jpeg.tobytes()

        # Yield the frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
detected_class_names = []

@app.route("/")
def index():
    return render_template('index.html',  detected_class_names=detected_class_names)

@app.route('/video')
def video():
    return Response(gen_camera(detected_class_names), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
