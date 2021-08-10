import cv2
import numpy as np

def scale_factor(x, y, w, h, alpha, size_image = (224, 224)):
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


img = cv2.imread("./Test/3.png")
orig_img = img.copy()
print(img.shape[0:2])

face_Cascade = cv2.CascadeClassifier("./Core/haarcascade_frontalface_default.xml")
eyesGlasses_Cascade = cv2.CascadeClassifier("./Core/haarcascade_eye_tree_eyeglasses.xml")

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

eyes = face_Cascade.detectMultiScale(gray_img, 1.01, 5)
print(type(eyes))

eyes_real = []
for (x, y, w, h) in eyes:
    roi_gray = gray_img[y:y+h, x:x+h]
    eye = face_Cascade.detectMultiScale(roi_gray, 1.01, 5)
    print(len(eye))
    if len(eye) >0:
        eyes_real.append([x, y, w, h])

for pos_eye in eyes_real:
    cv2.rectangle(img, (pos_eye[0], pos_eye[1]), (pos_eye[0] + pos_eye[2], pos_eye[1] + pos_eye[3]), (0, 255, 0), 2)
    print(img[pos_eye[1]:pos_eye[1] + pos_eye[3], pos_eye[0]:pos_eye[0] + pos_eye[2]].shape)

# faces = face_Cascade.detectMultiScale(gray_img, 1.01, 5)
# faces_real = []
# for (x, y, w, h) in faces:
#     roi_face_gray = gray_img[y:y+h, x:x+h]
#     face = face_Cascade.detectMultiScale(roi_face_gray, 1.05, 5)
#     if len(face) > 0:
#         faces_real.append([x, y, w, h])

# for pos_face in faces_real:
#     cv2.rectangle(img, (pos_face[0], pos_face[1]), (pos_face[0] + pos_face[2], pos_face[1] + pos_face[3]), (0, 255, 0), 2)

# roi_eyes_real = []
# eyes_real = []
# for face in faces_real:
#     roi_face = []
#     roi_gray_face = gray_img[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]
#     roi_color_face = orig_img[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]

#     eyes = eyesGlasses_Cascade.detectMultiScale(roi_gray_face, 1.01, 5)
#     face_size = roi_color_face.shape[0:2]        
#     for (x_, y_, w_, h_) in eyes:
#         roi_eyes_gray = roi_gray_face[y_:y_+h_, x_:x_+h_]
#         eye = eyesGlasses_Cascade.detectMultiScale(roi_eyes_gray, 1.01, 5)
#         if len(eye) > 0:
#             x1, y1, x2, y2 = scale_factor(x_, y_, w_, h_, alpha = 2, size_image = face_size)
#             roi_face.append(roi_color_face[y1:y2, x1:x2])
#             eyes_real.append([face[0]+x_, face[1]+y_, w_, h_])
    
#     roi_eyes_real.append(roi_face)

# for eyes_face in roi_eyes_real:
#     for eye_ in eyes_face:
#         cv2.imshow("Eyes",eye_)
#         cv2.waitKey()

for pos_eye in eyes_real:
    cv2.rectangle(img, (pos_eye[0], pos_eye[1]), (pos_eye[0] + pos_eye[2], pos_eye[1] + pos_eye[3]), (255, 0, 0), 2)

cv2.imshow("Test", img)
cv2.waitKey()
cv2.destroyAllWindows()