# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 14:43:38 2019

@author: wireshark
"""


#import numpy as np
#import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
from keras.utils import np_utils 
DATADIR = r"E:\tk\WORKING\alex_ocr\train_ocr"

CATEGORIES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
#IMG_SIZE=227

training_data = []

def create_training_data():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in os.listdir(path):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img))  # convert to array
#                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([img_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
               pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
                #print("general exception", e, os.path.join(path,img))

create_training_data()

print(len(training_data))

import random

random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])


X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.asarray(X)
y=np.asarray(y)
y=np_utils.to_categorical(y)


#----------------------------------------------------------------------------------

import keras
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import regularizers
np.random.seed(1000)

model=Sequential()

# CONV 1
model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))

# MaxPool 1
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2), padding='valid'))
model.add(BatchNormalization())

# CONV 2
model.add(Conv2D(filters=256, kernel_size=(11,11),strides=(1,1),padding='valid'))
model.add(Activation('relu'))

# Maxpool 2
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
model.add(BatchNormalization())

# CONV 3
model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())

# CONV 4
model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())

# CONV 5
model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='valid'))
model.add(Activation('relu'))

# MaxPool 3
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
model.add(BatchNormalization())

# Flattening model
model.add(Flatten())

# FullyConnected layer 1
model.add(Dense(4096,input_shape=(224*224*3,)))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())

# FullyConnected layer 2
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())

# FullyConnected layer 3
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())

# Final layer
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()
adam=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.1, amsgrad=False)

#sgd = optimizers.SGD(lr=0.01, decay=0.1, momentum=0.9, nesterov=True)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
#model.summary()
#model.fit(X,y,epochs=90,batch_size=128)





#adam = optimizers.SGD(lr=0.01, decay=0.1, momentum=0.9, nesterov=True)model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X,y,batch_size=128,epochs=40,verbose=1,validation_split=0.2,shuffle=True)


model.save('mnist_cnn.h5')






# model_1 = load_model('model_recent.h5') 
# print(model_1.summary())


# import cv2
# x = cv2.imread("1.png")
# ret,thresh2 = cv2.threshold(x,200,255,cv2.THRESH_BINARY_INV)
# final=cv2.resize(thresh2,(224,224))
# final_1 = final.reshape(1,224,224,3)
# plt.imshow(final,cmap='gray')
# np.argmax(model_1.predict(final_1))