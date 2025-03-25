import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras

model=keras.models.load_model("model.h5")
label=np.load("labels.npy")
data_size=0
holistic=mp.solutions.holistic 
hands=mp.solutions.hands
face_mesh=mp.solutions.face_mesh
holis=holistic.Holistic()
face=face_mesh.FaceMesh()
drawing=mp.solutions.drawing_utils
cap=cv2.VideoCapture(0)
X=[]
FACE_LANDMARKS = 468 
HAND_LANDMARKS = 21
COORDINATES_PER_LANDMARK = 2 
while True:
    lst=[]
    _, frm =cap.read()
    
    frm=cv2.flip(frm,1)
    
    res=holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
    face_res=face.process(cv2.cvtColor(frm,cv2.COLOR_BGR2RGB))
    
    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)
    else:                   
        lst.extend([0.0] * FACE_LANDMARKS * COORDINATES_PER_LANDMARK)
            
    if res.left_hand_landmarks:
        for i in res.left_hand_landmarks.landmark:
            lst.append(i.x -res.left_hand_landmarks.landmark[8].x)
            lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
    else:
        lst.extend([0.0] * HAND_LANDMARKS * COORDINATES_PER_LANDMARK)
            
    if res.right_hand_landmarks:
        for i in res.right_hand_landmarks.landmark:
            lst.append(i.x -res.right_hand_landmarks.landmark[8].x)
            lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
    else:
        lst.extend([0.0] * HAND_LANDMARKS * COORDINATES_PER_LANDMARK)

    X.append(lst)  
    data_size=data_size+1   
    lst=np.array(lst).reshape(1,-1)
    
    pred=np.argmax(model.predict(lst))
    print(pred)  
    cv2.putText(frm, str(label[pred]) ,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2) 
            
    
    if face_res.multi_face_landmarks:
        for landmarks in face_res.multi_face_landmarks:
            drawing.draw_landmarks(frm, landmarks, face_mesh.FACEMESH_TESSELATION)
    if res.left_hand_landmarks:
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    if res.right_hand_landmarks:
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)
    
    cv2.imshow("window",frm)
    
    if cv2.waitKey(1) != -1:
        cv2.destroyAllWindows()
        cap.release()
        break