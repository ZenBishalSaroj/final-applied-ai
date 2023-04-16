import cv2
from flask import Flask,render_template
import numpy as np
import tensorflow as tf
import keras.utils as image




app = Flask(__name__)

face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')




#################################### Landing ######################
@app.route('/')
def home():   
    return render_template('landing.html') 

def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points
    
@app.route('/monitor',methods=['GET'])
def start():
    model = tf.keras.models.load_model("best_model.h5")
    face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    h=False

    while True:
        ret, test_img = cap.read()  
        if not ret:
            continue
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
            roi_gray = gray_img[y:y + w, x:x + h]  
            roi_gray = cv2.resize(roi_gray, (224, 224))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            predictions = model.predict(img_pixels)

            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Facial emotion analysis ', resized_img)

        if cv2.waitKey(1)==27:
            break

    cap.release()
    cv2.destroyAllWindows
    return render_template('monitor.html') 


######################### Info ##########################
@app.route('/info')
def info_index():
    return render_template('info.html')




if __name__ == '__main__':
    app.run(debug=True)