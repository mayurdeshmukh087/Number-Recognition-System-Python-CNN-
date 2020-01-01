import tkinter as tk
from tkinter import filedialog
from tkinter import *
import os
import cv2
import PIL.Image, PIL.ImageTk

import keras
from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

model=load_model(r'C:\Users\Paresh\Desktop\shankar\2016BEC066 & 2016BEC155\Model & Program\mnist_cnn.h5')

def res(path):
    final=cv2.imread(r'{}'.format(path))
    final=cv2.resize(final,(28,28))
    final=cv2.cvtColor(final,cv2.COLOR_BGR2GRAY)
    final=final.reshape(1,28,28,1)
    pred=np.argmax(model.predict(final))
    printSomething(pred)
    print(pred)
    

def loader(groot):
    im = PIL.Image.open(groot)
    im = im.resize((420, 280), PIL.Image.ANTIALIAS)
    tkimage=PIL.ImageTk.PhotoImage(im)
    myvar=Label(root,image=tkimage)
    myvar.image=tkimage
    myvar.place(x=10,y=180)

def printSomething(myCmd):
    l4 = tk.Label(root, text=myCmd, fg = "green", justify=tk.LEFT, font = ("Times bold italic",40)).place(x=600,y=380)

def image():
    l4=tk.Label(root, text="                                    ", fg = "green", justify=tk.LEFT, font = ("Times bold italic",20)).place(x=450,y=420)
    root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    loader(root.filename)
    print('yo',root.filename)
    res(root.filename)

root = tk.Tk()
root.title("Number Classifier    [Rohan Todkar Mayur Deshmukh]")
canvas = tk.Canvas(root, width = 800, height = 500)
canvas.pack()

l1=tk.Label(root, text="IT Department", fg = "blue", justify=tk.LEFT, font = ("Times",18)).place(x=330,y=50)
l2=tk.Label(root, text="Shri Guru Gobind Singhji Institute of Engineering and Technology, Nanded", fg = "red", justify=tk.LEFT, font = ("Times",15)).place(x=20,y=0)
l3=tk.Label(root, text="Demo for Number Classifier", fg = "green", justify=tk.LEFT, font = ("Times bold italic",20)).place(x=180,y=85)
l4=tk.Label(root, text="  ", fg = "green", justify=tk.LEFT, font = ("Times bold italic",20)).place(x=450,y=420)

frame = tk.Frame(root)
frame.pack() 

btn_1 = tk.Button(root, text="Load Image", width=10, height=1, fg="purple", command=image).place(x=600,y=230)

root.mainloop()