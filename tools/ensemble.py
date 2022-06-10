import cv2
import os
import numpy as np
files = [
# "./resnet101_finetune16_1.csv",
# "./resnet101_finetune16_2.csv",
# "./resnext101_finetune16_1.csv",
# "./resnext101_finetune16_2.csv",
# "./resnext101_finetune20_2.csv",
# "./resnext101_finetune20_3.csv",
# "./resnext101_finetune20.csv",
# "./resnext101_finetune100.csv",
"./resnext101_1_con_56.csv",
"./resnext101_1_56.csv",
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
with open("./ensemble.csv", "w") as f1:
    f1.write("imageA,imageB,prediction\n")
    for img_path1, img_path2, ensemble_score in zip(img_paths1, img_paths2,scores_list):
        # f1.write("{},{},{}\n".format(img_path1,img_path2,1-ensemble_score))
        f1.write("{},{},{}\n".format(img_path1,img_path2,ensemble_score))