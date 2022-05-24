from distutils import filelist
from fileinput import filelineno
from shutil import copyfile
import numpy as np
import cv2
import os
from matplotlib import pyplot as plot
import random,shutil

from pytest import Testdir

# original = '/Users/garylin/Desktop/開放資料/AI Cup/data/sugarcane/20180921-2-0142.JPG'
# after = '/Users/garylin/Desktop/開放資料/AI Cup/data'

# img_org = cv2.imread(original)
# img_aft1 = cv2.resize(img_org, (1024,1024), interpolation = cv2.INTER_AREA)
# img_aft2 = cv2.resize(img_org, (512,512), interpolation = cv2.INTER_AREA)


# title = ['Original', 'Resize1024', 'Resize512']
# imgs = [img_org, img_aft1, img_aft2]
# for i in range(3):
#     plot.subplot(1,3,i+1)
#     plot.imshow(imgs[i])
#     plot.title(title[i])
#     plot.xticks([]), plot.yticks([])
#     cv2.imwrite(after+str(i)+'.jpg',imgs[i])
# plot.show()

#Resize 影像（全資料夾）
# path = "D://User//Desktop//AI_CUP//Final_Test//Question"
# store_path = "D://User//Desktop//AI_CUP//Final_Test//Question512"
# if not os.path.exists(store_path):
#             os.mkdir(store_path)

# for folder in os.listdir(path):
#     if not os.path.exists(store_path + '//' + folder):
#         os.mkdir(store_path + '//' + folder)
#     for img_file in os.listdir(os.path.join(path, folder)):
#         directory = store_path + '//' + folder + '//' + img_file
#         print(directory)
#         img = cv2.imread(os.path.join(path, folder, img_file))
#         img = cv2.resize(img, (512,512), interpolation = cv2.INTER_AREA)
#         cv2.imwrite(directory, img)


#Resize 影像（單一資料夾）
# path = '/Users/garylin/Desktop/開放資料/AI Cup/data/tomato'

# for img_file in os.listdir(path):
#     directory = os.path.join(path, img_file)
#     print(os.path.join(path, img_file))
#     img = cv2.imread(os.path.join(path, img_file))
#     img = cv2.resize(img, (512,512), interpolation = cv2.INTER_AREA)
#     cv2.imwrite(directory, img)

#抓取特定數量照片
# path = "D://User//Desktop//AI_CUP//Train_data//Test1000-512"
# store_path = "D://User//Desktop//AI_CUP//Train_data//Test2000-512"
# if not os.path.exists(store_path):
#     os.mkdir(store_path)

# for folder in os.listdir(path):
#     if not os.path.exists(store_path + '//' + folder):
#         os.mkdir(store_path + '//' + folder)
#     i = 1
#     for img_file in os.listdir(os.path.join(path, folder)):
#         directory = store_path + '//' + folder + '//' + img_file
#         print(directory)
#         img = cv2.imread(os.path.join(path, folder, img_file))
#         cv2.imwrite(directory, img)
#         if i >= 2000:
#              break
#         i += 1


#隨機抽取特定數量圖片
def CopyFile(fileDir):
    if not os.path.exists(tarDir):
        os.mkdir(tarDir)
    for folder in os.listdir(fileDir):
        if not os.path.exists(tarDir+'//'+folder):
            os.mkdir(tarDir+'//'+folder)
        pathDir = os.path.join(fileDir, folder)
        pathDir = os.listdir(pathDir)
        filenumber = len(pathDir)
        print(filenumber)
        picknumber = 2000
        if filenumber < picknumber:
            picknumber = filenumber
        sample = random.sample(pathDir, picknumber)
        for name in sample:
            shutil.copy(fileDir+'//'+folder+'//'+name, tarDir+'//'+folder+'//'+name)
            print(tarDir+'//'+folder+'//'+name)
    
    return

#隨機抽取特定數量特定資料夾圖片
# def CopyFile(fileDir):
#     if not os.path.exists(tarDir):
#         os.mkdir(tarDir)
#     pathDir = os.listdir(fileDir)
#     filenumber = len(pathDir)
#     print(filenumber)
#     picknumber = 50
#     if filenumber < picknumber:
#         picknumber = filenumber
#     sample = random.sample(pathDir, picknumber)
#     print(sample)
#     for name in sample:
#         shutil.copy(fileDir+'//'+name, tarDir+'//'+name)
#     # print(tarDir+'//'+name)
    
#     return

#隨機抓取測試檔案
def TestDataset(fileDir):
    if not os.path.exists(testDir):
        os.mkdir(testDir)
    if not os.path.exists(valDir):
        os.mkdir(valDir)
    for folder in os.listdir(fileDir):
        if not os.path.exists(os.path.join(testDir, folder)):
            os.mkdir(os.path.join(testDir, folder))
        if not os.path.exists(os.path.join(valDir, folder)):
            os.mkdir(os.path.join(valDir, folder))
        Filelist = os.listdir(os.path.join(fileDir, folder))
        print(folder)
        # print(Filelist)
        print(len(Filelist), type(Filelist))

        Tarlist = os.listdir(os.path.join(tarDir, folder))
        print(len(Tarlist), type(Tarlist))
        
        Vallist = []
        picknumber = 150
        if len(Tarlist)+picknumber <= len(Filelist):
            for i in Filelist:
                for j in Tarlist:
                    if i == j:
                        break
                else:
                    Vallist.append(i)
            print(len(Vallist))

        else:
            continue
        
        sample = random.sample(Vallist, picknumber)
        for name in sample:
            shutil.copy(fileDir+'//'+folder+'//'+name, valDir+'//'+folder+'//'+name)
        
        Vallist = os.listdir(os.path.join(valDir, folder))
        Testlist = []
        picknumber = 150
        if picknumber+len(Vallist)+len(Tarlist) <= len(Filelist):
            for i in Filelist:
                x = False
                for j in Tarlist:
                    for k in Vallist:
                        if i == j or i == k:
                            x = True
                            break
                    if x == True:
                        break
                else:
                    Testlist.append(i)
            print(len(Testlist))
        
        else:
            continue

        sample = random.sample(Testlist, picknumber)
        for name in sample:
            shutil.copy(fileDir+'//'+folder+'//'+name, testDir+'//'+folder+'//'+name)
        

    return

if __name__ == '__main__':
    fileDir = "D://User//Desktop//AI_CUP//Train_data//enhance"
    tarDir = "D://User//Desktop//AI_CUP//Train_data//enhance2200"
    valDir = "D://User//Desktop//AI_CUP//Test_data//Val150"
    testDir = "D://User//Desktop//AI_CUP//Test_data//Test150"
    CopyFile(fileDir)
    # TestDataset(fileDir)