import cv2
import os
import numpy as np
files = [
"./submit1.csv",
"./submit2.csv",
"./submit3.csv",
"./submit4.csv",
"./submit5.csv",
"./submit6.csv",
"./submit7.csv",
"./submit8.csv",
]
scores_list=[]
for file in files:
   with open(file, "r") as f1:
    data = f1.readlines()
    data = data[1:]
    img_paths1,img_paths2,scores = [],[],[]
    for line in data:
        img_path1, img_path2, score = str(line.strip()).split(',')
        scores.append(float(score))
        img_paths1.append(img_path1)
        img_paths2.append(img_path2)
    scores_list.append(scores)
    print(len(scores))
scores_list = np.array(scores_list)
# print(scores_list)
scores_list = np.mean(scores_list,axis=0)
# print(scores_list.shape)
with open("./submit.csv", "w") as f1:
    f1.write("imageA,imageB,prediction\n")
    for img_path1, img_path2, ensemble_score in zip(img_paths1, img_paths2,scores_list):
        # f1.write("{},{},{}\n".format(img_path1,img_path2,1-ensemble_score))
        f1.write("{},{},{}\n".format(img_path1,img_path2,ensemble_score))