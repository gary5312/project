from IPython.lib.display import FileLinks
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout,LeakyReLU
from tensorflow.keras import utils,optimizers
import numpy as np
import cv2
import csv
from sklearn.utils import shuffle
import matplotlib.pyplot as plot
import os,math,time
from random import randint
from keras.callbacks import Callback
import keras.callbacks as callbacks
from keras import backend as K

model1_dir = '/content/drive/MyDrive/AICUP-G/model/EFF0505.h5'
model2_dir = '/content/drive/MyDrive/AICUP-G/model/c_s_p.h5'


def class_list():
    land_list = ['bareland','peanut','guava','carrot','pineapple','banana','dragonfruit','garlic','corn',
    'pumpkin','sugarcane','rice','soybean','tomato']
    return land_list

def special_class_list():
    land_list = ['carrot','soybean','peanut']
    return land_list

def get_class(label_code):
    classes = class_list()
    return classes[label_code]

def get_special_class(label_code):
    classes = special_class_list()
    return classes[label_code]

def get_img(path):
    Images = []
    Names = []
    start = time.time()
    for img_file in os.listdir(path):
        image = cv2.imread(os.path.join(path, img_file))
        image = cv2.resize(image,(224,224))
        Images.append(image)
        Names.append(img_file)
    end = time.time()
    print('Finish time :{:.2f}s'.format(end-start))

    return (Images, Names)

def creat_csv(info):
    path = '/content/drive/MyDrive/AICUP-G/Final_Test/2model_1644.csv'
    with open(path, 'a+')as csvfile:
        csv_write = csv.writer(csvfile)
        csv_write.writerow(info)

test_dir = '/content/drive/MyDrive/AICUP-G/Final_Test/Question512'
# test_images, test_names = get_img(test_dir)
# test_images = np.array(test_images)
# print(test_images.shape)

model = load_model(model1_dir)
enhance_model = load_model(model2_dir)


for folder in os.listdir(test_dir):
    x = 0
    print('Start:', folder)
    for img_file in os.listdir(os.path.join(test_dir, folder)):
        image = cv2.imread(os.path.join(test_dir, folder, img_file))
        image = cv2.resize(image,(224,224))
        image = image.reshape(1,224,224,3)
        prediction = model(image, training = False)
        prediction = np.argmax(model(image, training = False), axis = 1)
        info = img_file, get_class(prediction[0])

        if get_class(prediction[0]) == 'carrot' or get_class(prediction[0]) == 'soybean' or get_class(prediction[0]) == 'peanut':
            prediction = enhance_model(image, training = False)
            prediction = np.argmax(prediction, axis = 1)
            info = img_file, get_special_class(prediction[0])
    
        creat_csv(info)
        print(x,',',info)
        x += 1
    