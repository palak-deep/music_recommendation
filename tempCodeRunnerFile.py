import os
import numpy as np
import cv2
from tensorflow import keras

# from tensorflow.keras.utils import to_categorical
# from keras.layers import Input , Dense
# from keras.models import Model


is_init=False
size=-1
label=[]
dictionary={}
c=0
for i in os.listdir():
    if i.split(".")[-1] == "npy":
        current_data = np.load(i)
        current_size = current_data.shape[0]
        current_labels = np.array([i.split('.')[0]] * current_size).reshape(-1, 1)

        if not is_init:
            is_init = True
            X = current_data
            y = current_labels
        else:
            X = np.concatenate((X, current_data), axis=0)
            y = np.concatenate((y, current_labels), axis=0)

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c
        c += 1

# print(X)
# print(y)            
      
# print(dictionary)
# print(label)  
# print(y)
for i in range(y.shape[0]):
    y[i,0]=dictionary[y[i, 0]]
y = np.array(y, dtype="int32")    

y =keras.utils.to_categorical(y)

X_new =X.copy()
y_new=y.copy()
counter=0
cnt=np.arange(X.shape[0])  #provides a list 
np.random.shuffle(cnt)

for i in cnt:
    X_new[counter] = X[i]
    y_new[counter] = y[i]
    counter=counter+1


ip = keras.layers.Input(shape=(X.shape[1],))  # Add a comma



m=keras.layers.Dense(512 , activation ="relu")(ip)
n=keras.layers.Dense(256 , activation ="relu")(m)

op=keras.layers.Dense(y.shape[1], activation='softmax')(m)
model =keras.models.Model(inputs =ip , outputs=op)

model.compile(optimizer='rmsprop' , loss='categorical_crossentropy',metrics=['acc'])
model.fit(X , y ,epochs=50)
model.save("model.h5")
np.save('labels.npy' , np.array(label))


