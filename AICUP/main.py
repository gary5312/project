from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout,LeakyReLU
from tensorflow.keras import utils,optimizers
import numpy as np
import cv2
from sklearn.utils import shuffle
import matplotlib.pyplot as plot
import os,math,time
from random import randint
from keras.callbacks import Callback,ModelCheckpoint
import keras.callbacks as callbacks
from keras import backend as K
from datetime import datetime
#訓練網路
def class_list():
    land_list = ['bareland','inundated','peanut','guava','carrot','pineapple','banana','dragonfruit','garlic','corn',
    'pumpkin','sugarcane','rice','soybean','tomato']
    return land_list

def get_images(path):
    Images = []
    Lables = []
    label = 0
    classes = class_list()
    for folder in os.listdir(path):
        start = time.time()
        label = classes.index(folder)
        for img_file in os.listdir(os.path.join(path, folder)):
            image = cv2.imread(os.path.join(path, folder, img_file))
            image = cv2.resize(image,(224,224))#resize尺寸大小
            Images.append(image)
            Lables.append(label)
        end = time.time()
        print('Finish folder:',folder)
        print('time used:{:.2f}s'.format(end-start))
    
        
    return shuffle(Images,Lables)

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
    


class SnapshotEnsemble(Callback):
    def __init__(self, n_epochs, n_cycles, lrate_max, verbose = 0):
        self.epochs = n_epochs
        self.cycles = n_cycles
        self.lr_max = lrate_max
        self.lrates = list()

    def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
        epochs_per_cycle = n_epochs // n_cycles
        cos_inner = (np.pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
        return lrate_max/2 * (np.cos(cos_inner) + 1)

    def on_epoch_begin(self, epoch, logs={}):
        lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)
        print(f'epoch {epoch+1}, lr {lr}')
        K.set_value(self.model.optimizer.lr, lr)
        self.lrates.append(lr)

    def on_epoch_end(self, epoch, logs={}):
        epochs_per_cycle = n_epochs // n_cycles

train_path = '/content/drive/MyDrive/AICUP-G/Random2000'
train_images, train_labels = get_images(train_path)

train_images = np.array(train_images)
train_labels = np.array(train_labels)

print("Shpae of Images:",train_images.shape)
print("Shape of Labels:",train_labels.shape)

plot_img(train_images, train_labels)

#VGG16 model
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', input_shape=(224,224,3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(rate=0.5)) 
model.add(Dense(4096, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(15, activation='softmax')) 

opt = optimizers.Adam(learning_rate=0.00001)
model.compile(loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'], optimizer= opt)
# model.compile('Adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.summary()

n_epochs = 75


#CheckPoint
filepath='/content/drive/MyDrive/AICUP-G/model/best_model.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only = False, mode='max')
callbacks_list = [checkpoint]

# train_history = model.fit(x = train_images, y = train_labels, validation_split = 0.2, epochs = n_epochs, batch_size = 64, callbacks = [ca])

show_train_history(train_history, 'accuracy', 'val_accuracy')
show_train_history(train_history, 'loss', 'val_loss')

# scores = model.evaluate(train_images, train_labels)
# print(scores)
#測試網路
