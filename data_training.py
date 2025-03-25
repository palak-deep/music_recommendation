import os
import numpy as np
import cv2
from tensorflow import keras

is_init = False
label = []
dictionary = {}
c = 0

for i in os.listdir():
    if i.endswith(".npy"):
        current_data = np.load(i)
        
        # Ensure data is at least 2D
        if current_data.ndim == 1:  
            current_data = current_data.reshape(-1, 1)  

        current_size = current_data.shape[0]
        current_labels = np.array([i.split('.')[0]] * current_size).reshape(-1, 1)

        if not is_init:
            is_init = True
            X = current_data
            y = current_labels
        else:
            if current_data.shape[1] != X.shape[1]:  
                print(f"Dimension mismatch in {i}: {current_data.shape[1]} vs {X.shape[1]}")
                continue  # Skip inconsistent files
            
            X = np.concatenate((X, current_data), axis=0)
            y = np.concatenate((y, current_labels), axis=0)

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c
        c += 1

# Convert labels to categorical
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")    
y = keras.utils.to_categorical(y)

# Shuffle Data
X_new = X.copy()
y_new = y.copy()
counter = 0
cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

for i in cnt:
    X_new[counter] = X[i]
    y_new[counter] = y[i]
    counter += 1

# Model Definition
ip = keras.layers.Input(shape=(X.shape[1],))  
m = keras.layers.Dense(512, activation="relu")(ip)
n = keras.layers.Dense(256, activation="relu")(m)
op = keras.layers.Dense(y.shape[1], activation='softmax')(m)

model = keras.models.Model(inputs=ip, outputs=op)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
model.fit(X_new, y_new, epochs=50)

model.save("model.h5")
np.save('labels.npy', np.array(label))
