from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout,LeakyReLU
from tensorflow.keras import utils,optimizers
import numpy as np
import cv2
from sklearn.utils import shuffle
import matplotlib.pyplot as plot
import os,math,time
from random import randint
from keras.callbacks import Callback
import keras.callbacks as callbacks
from keras import backend as K
from datetime import datetime

def class_list():
    land_list = ['bareland','peanut','guava','carrot','pineapple','banana','dragonfruit','garlic','corn',
    'pumpkin','sugarcane','rice','soybean','tomato']
    return land_list

# def class_list():
#     land_list = ['carrot','soybean','peanut (1)']
#     return land_list

def get_images(path):
    Images = []
    Labels = []
    label = 0
    classes = class_list()
    for folder in os.listdir(path):
        start = time.time()
        label = classes.index(folder)
        for img_file in os.listdir(os.path.join(path, folder)):
            image = cv2.imread(os.path.join(path, folder, img_file))
            image = cv2.resize(image,(224,224))#resize尺寸大小
            Images.append(image)
            Labels.append(label)
        end = time.time()
        print('Finish folder:',folder)
        print('time used:{:.2f}s'.format(end-start))
    
        
    return shuffle(Images,Labels)

def single_folder(path):
    Images=[]
    Labels=[]
    label='bareland'
    for img_file in os.listdir(path):
        image=cv2.imread(os.path.join(path, img_file))
        image=cv2.resize(image,(224,224))
        Images.append(image)
        Labels.append(label)

    return shuffle(Images,Labels)

def get_class(label_code):
    classes = class_list()
    return classes[label_code]

def plot_img(Images, Labels, Prediction = [], num=12):
    if num>30: num = 30
    row = math.ceil(num/6)
    fig, ax = plot.subplots(row, 6, figsize=(24, row*3))
    pic = 1
    for i in range(row):
        for j in range(6):
            if pic <= num:
                idx = randint(0, len(Images)-1)
                ax[i,j].imshow(Images[idx])
                title = get_class(Labels[idx])
                if len(Prediction) > 0:
                    title = "gt={},p={}".format(title,get_class(Prediction[idx]))
                ax[i,j].set_title(title)
                
        ax[i,j].axis('off')
        pic += 1                                            
    plot.show()
                  

def show_train_history(train_history, train, validation):  
    plot.plot(train_history.history[train])
    plot.plot(train_history.history[validation])
    plot.title('Train History')  
    plot.ylabel(train)  
    plot.xlabel('Epoch')  
    plot.legend(['train', 'validation'], loc='upper left')  
    plot.show()

model_dir = '/content/drive/MyDrive/AICUP-G/model/EFFL0512Best.h5'
model = load_model(model_dir)
print(model_dir)

test_dir = '/content/drive/MyDrive/AICUP-G/Test_data/Test150'
# test_images, test_labels = single_folder(test_dir)
test_images, test_labels = get_images(test_dir)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

print("Shape of Images:",test_images.shape)
print("Shape of Labels:",test_labels.shape)

score = model.evaluate(test_images, test_labels)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# prediction_prob = model.predict(test_images)
# prediction = np.argmax(prediction_prob,axis=1)
# print('Prediction:', prediction_prob[0:20])
# print('Actual:', test_labels)

# print("gt:ground truth, p: prediction\n")
# plot_img(test_images, test_labels, prediction, 30)

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
import pandas as pd
def print_classification_report(x,y):
# Confution Matrix and Classification Report

        
    Y_pred = model.predict(x)
    y_pred = np.argmax(Y_pred, axis=1)

    print('Classification Report')
    print(classification_report(y, y_pred, target_names=class_list()))

    print('Confusion Matrix')
    conf_mat = confusion_matrix(y, y_pred)
    df_cm = pd.DataFrame(conf_mat, index=class_list(), columns=class_list())
    plot.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)

print_classification_report(test_images, test_labels)