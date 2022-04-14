import os

path = '/content/drive/MyDrive/2022_H1_project/0305colab_only/darknet/ts/ts'
for filename in os.listdir(path):
    # print(os.path.join(path,filename))
    x = os.path.join(path,filename)
    if x[-1] == 'g':
        print(x)
        f = open("all1.txt","a")
        f.write(x+"\n")
        f.close()

print('finish')

import sys
result=[]
with open("all1.txt","r") as f:
    for line in f:
        result.append(list(line.strip("\n").split(',')))
# print(result[630])

for i in range(630):
    f = open("train1.txt","a")
    f.writelines(result[i])
    f.write("\n")
    f.close()


for i in range(630,741):
    f = open("test1.txt","a")
    f.writelines(result[i])
    f.write("\n")
    f.close()

# path = 'test.txt'
# x = 0
# with open(path) as f:
#     lines = f.readlines()
#     for line in lines:
#         print(line)
#         x += 1
#         print(x)