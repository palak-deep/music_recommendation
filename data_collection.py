import mediapipe as mp
import numpy as np
import cv2

cap=cv2.VideoCapture(0)
#it will capture the vedio using our computers web cam
#setting up mediapipe
name=input("Enter the name of the data : ")
holistic=mp.solutions.holistic 
#holistic is used to detect the whole body
hands=mp.solutions.hands
#this is used to detect the hand landmarks
face_mesh=mp.solutions.face_mesh
#this is used to detect the face landmarks and to join these landmarks
holis=holistic.Holistic()
#holis is the instance of holistic to detect the body
face=face_mesh.FaceMesh()
#face is used to initialize the Facemesh that will detect whole facial landmarks.
drawing=mp.solutions.drawing_utils
#this utility helps to draw landmarks and connections on image.
X=[] # empty list where we will store the detected data.(landmarks)
data_size=0 #this keep track how many images we had processed in web cam.
FACE_LANDMARKS = 468  # 468 landmarks for the face mesh
HAND_LANDMARKS = 21  # 21 landmarks per hand (left and right hand)
COORDINATES_PER_LANDMARK = 2  # Each landmark has 2 coordinates (x, y)
while True:
    lst=[]
    _, frm =cap.read()
    #it will capture the frame from the cam._is used because we dont want the first value that is true or false indicating vedio is captured successfully or not.frm holds the actual value.
    
    frm=cv2.flip(frm,1)
    #it is used to flip the captured image horizontally as an mirror image.
    
    res=holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
    face_res=face.process(cv2.cvtColor(frm,cv2.COLOR_BGR2RGB))
    # it will convert the default BGR color(used by cv2) to RGB color (used by mediapipe)
    
    if res.face_landmarks:# check if face is detected or not
        for i in res.face_landmarks.landmark:#it will iterate upto 468 points becuse face_mesh break face into 468 landmarks.
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
        
            
    
    if face_res.multi_face_landmarks:
        for landmarks in face_res.multi_face_landmarks:
            drawing.draw_landmarks(frm, landmarks, face_mesh.FACEMESH_TESSELATION)
            #drawing.draw_landmarks()is a function that draw landmarks in the frame.it carry three parameters.frm:the image from the webcam that we are processing.landmarks:actual keypoints that are detected on the face.FACEMESH_TESSELATION:it connects the key points or landmarks that showing the mesh or grid structure on the face.
    if res.left_hand_landmarks:
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    if res.right_hand_landmarks:
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)
        
    cv2.putText(frm,str(data_size) ,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    #this is used to put text on the image it includes parameters=> frm:frame captured from webcam.str(data_size):is used to keep track of how many frames had been processed .it will show output in string becouse of typecasting.(50,50): is used for centre it will take 50 from vertical and 50 from horizontal side.cv2.FONT_HERSEY_SIMPLEX:is the simplest font in the cv2 for displaying text.1:is the font size.(0,255,0):is the color 0 for blue,255 for green ,0 for red color.2:for thickness of the text
    
    cv2.imshow("window",frm)
    #this shows the captured frame in the new window named window.
    
    if cv2.waitKey(1) ==27 or data_size>90:
        #it is used to detect if any key is pressed or not.(1)is used to detect if any key is pressed within 1 milisecond.27 is the escape key.
        cv2.destroyAllWindows()
        #if escape is pressed it will close the window which is showing the web cam.
        cap.release()
        #it will release the cam telling the computer thet its done.
        break
    
np.save(f"{name}.npy",np.array(X))
print(np.array(X).shape)