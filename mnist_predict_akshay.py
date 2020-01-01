# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 00:28:11 2019

@author: wireshark
"""

import keras
from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

model=load_model(r'C:\Users\Paresh\Desktop\shankar\2016BEC066 & 2016BEC155\Model & Program\mnist_cnn.h5')

final=cv2.imread(r'C:\Users\Paresh\Desktop\shankar\2016BEC066 & 2016BEC155\Test\qw.png')
final=cv2.resize(final,(28,28))
final=cv2.cvtColor(final,cv2.COLOR_BGR2GRAY)
# plt.imshow(final)
final=final.reshape(1,28,28,1)
pred=np.argmax(model.predict(final))
print(pred)