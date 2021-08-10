import os
import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from threading import Thread

EYE_AR_CONSEC_FRAMES = 30
eyes_status = False
mouth_status = False
saying = False
COUNTER = 0


class Face_Detector():
    def __init__(self, face_cascade_path, eye_cascade_path):
        self.face_Cascade = cv2.CascadeClassifier(face_cascade_path)
        self.eyes_Cascade = cv2.CascadeClassifier(eye_cascade_path)


    def detect_face_eyes(self, img, draw_face = True, draw_eyes = True):
        orig_img = img.copy()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_Cascade.detectMultiScale(gray_img, 1.01, 5)
        faces_real = []
        for (x, y, w, h) in faces:
            roi_face_gray = gray_img[y:y+h, x:x+h]
            face = self.face_Cascade.detectMultiScale(roi_face_gray, 1.01, 5)
            if len(face) > 0:
                faces_real.append([x, y, w, h])
        if draw_face:
            for pos_face in faces_real:
                cv2.rectangle(img, (pos_face[0], pos_face[1]), (pos_face[0] + pos_face[2], pos_face[1] + pos_face[3]), (0, 255, 0), 2)

        roi_eyes_real = []
        eyes_real = []
        for face in faces_real:
            roi_face = []
            roi_gray_face = gray_img[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]
            roi_color_face = orig_img[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]

            eyes = self.eyes_Cascade.detectMultiScale(roi_gray_face, 1.01, 5)
            face_size = roi_color_face.shape[0:2]        
            for (x_, y_, w_, h_) in eyes:
                roi_eyes_gray = roi_gray_face[y_:y_+h_, x_:x_+h_]
                eye = self.eyes_Cascade.detectMultiScale(roi_eyes_gray, 1.01, 5)
                if len(eye) > 0:
                    x1, y1, x2, y2 = self.scale_factor(x_, y_, w_, h_, alpha = 2, size_image = face_size)
                    roi_face.append(roi_color_face[y1:y2, x1:x2])
                    eyes_real.append([face[0]+x_, face[1]+y_, w_, h_])
            
            roi_eyes_real.append(roi_face)

        if draw_eyes:
            for pos_eye in eyes_real:
                cv2.rectangle(img, (pos_eye[0], pos_eye[1]), (pos_eye[0] + pos_eye[2], pos_eye[1] + pos_eye[3]), (255, 0, 0), 2)

        return img, roi_eyes_real


    def scale_factor(self, x, y, w, h, alpha, size_image = (224, 224)):
        new_x_tl = int(x + w*(1-alpha)/2)
        if new_x_tl < 0:
            new_x_tl = 0
        new_y_tl = int(y + h*(1-alpha)/2)
        if new_y_tl < 0:
            new_y_tl = 0
        new_x_br = int(x + w*(1+alpha)/2)
        if new_x_br > size_image[0]:
            new_x_br = size_image[0]        
        new_y_br = int(y + h*(1+alpha)/2)        
        if new_y_br > size_image[1]:
            new_y_br = size_image[1]
        return new_x_tl, new_y_tl, new_x_br, new_y_br

class Eyes_Detector():
    def __init__(self, model, file_weights_path):
        self.model = model
        self.file_weights_path = file_weights_path
        self.model.load_weights(self.file_weights_path)

    def preprocess_img(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        back2rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        img_pred = cv2.resize(back2rgb, (224, 224))
        return img_pred

    def predict_image(self,img):
        img_pred = self.preprocess_img(img)
        X_input = np.array(img_pred).reshape(1, 224, 224, 3)

        X_input = X_input/255.0

        prediction = self.model.predict(X_input)

        if prediction >= 0.5:
                out = "Open"
        else:
                out = "Close" 

        return out

def alarm(msg):
    global eyes_status
    global mouth_status
    global saying

    while eyes_status:
        print('call')
        s = 'espeak "'+msg+'"'
        os.system(s)

    if mouth_status:
        print('call')
        saying = True
        s = 'espeak "' + msg + '"'
        os.system(s)
        saying = False



def main():
    global EYE_AR_CONSEC_FRAMES
    global eyes_status
    global mouth_status
    global saying
    global COUNTER

    print("Loading Model MobileNet...")
    model = tf.keras.applications.MobileNet()

    base_input = model.layers[0].input
    base_output = model.layers[-4].output
    Flat_layer = layers.Flatten()(base_output)
    final_output = layers.Dense(1)(Flat_layer)
    final_output = layers.Activation('sigmoid')(final_output)
    model = Model(inputs = base_input, outputs = final_output)

    detect_blink_eye = Eyes_Detector(model = model, file_weights_path = "./Core/mobileNet_Model.h5")
    print("Finish Loading Model MobileNet!!!")

    print("Loading Model Eyes - Face Detection...")
    detect_eyes_in_face = Face_Detector(face_cascade_path = "./Core/haarcascade_frontalface_default.xml", eye_cascade_path = "./Core/haarcascade_eye_tree_eyeglasses.xml")
    print("Finished Load Model Eyes - Face Detection")

    pTime = 0
    cap = cv2.VideoCapture(0)
    time.sleep(1.0)


    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        orig_frame = frame.copy()
        frame, face_eyes = detect_eyes_in_face.detect_face_eyes(frame, draw_face = True, draw_eyes = True)

        if len(face_eyes) ==0:
            cv2.putText(frame, "Cannot find any face!!!", (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)            

        for i, face in enumerate(face_eyes):
            eyes_state = []
            for j, eye in enumerate(face):
                predict = detect_blink_eye.predict_image(eye)
                eyes_state.append(predict)
                print(f"On face {i+1}, eye {j+1} is " + predict)

            if "Open" not in eyes_state:
                COUNTER += 1

                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if eyes_status == False:
                        eyes_status = True
                        t = Thread(target=alarm, args=('wake up, wake up, wake up sir',))
                        t.deamon = True
                        t.start()

                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                eyes_status = False

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)

        cv2.imshow("Orig Video", orig_frame)
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.realease()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()