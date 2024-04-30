import os
import PIL
import cv2
import glob
import numpy as np
from tkinter import *
from PIL import Image, ImageDraw, ImageGrab
from keras.models import Sequential
from keras.layers import Dense, Flatten

from tensorflow.keras import layers
from tensorflow.keras import models
from keras.datasets import mnist
from keras.utils import to_categorical

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.load_weights('mnist.h5')

def event_function(event):
    
    x=event.x
    y=event.y
    
    x1=x-30
    y1=y-30
    
    x2=x+30
    y2=y+30

    canvas.create_oval((x1,y1,x2,y2),fill='black')
    img_draw.ellipse((x1,y1,x2,y2),fill='white')
    
def save():
    global count
    
    img_array=np.array(img)
    img_array=cv2.resize(img_array,(28,28))
    
    cv2.imwrite(str(count)+'.jpg',img_array)
    count=count+1
    
    
def predict():
    
    img_array=np.array(img)
    img_array=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
    img_array=cv2.resize(img_array,(28,28))
    
    img_array=img_array/255.0
    img_array=img_array.reshape(1,28,28)
    result=model.predict(img_array)
    label=np.argmax(result,axis=1)
    
    label_status.config(text='PREDICTED DIGIT:'+str(label))
    
count = 0

win = Tk()
canvas = Canvas(win, width=500, height=500, bg='white')
canvas.grid(row=0, column=0, columnspan=4)

button_predict = Button(win, text='PREDICT', bg='blue', fg='white', font='Helvetica 20 bold', command=predict)
button_predict.grid(row=1, column=1)

button_clear = Button(win, text='CLEAR', bg='yellow', fg='white', font='Helvetica 20 bold', command=clear)
button_clear.grid(row=1, column=2)

label_status = Label(win, text='PREDICTED DIGIT: NONE', bg='white', font='Helvetica 24 bold')
label_status.grid(row=2, column=0, columnspan=4)

canvas.bind('<B1-Motion>', event_function)
img = Image.new('RGB', (500, 500), (0, 0, 0))
img_draw = ImageDraw.Draw(img)

win.mainloop()
