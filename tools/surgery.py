from cv2 import sort
import numpy as np
import json

cluster=0
index_cluster={}
with open("./ensemble_avgmodels.csv", "r") as f1:
    data = f1.readlines()
    data = data[1:]
    img_paths1,img_paths2,scores = [],[],[]
    for line in data:
        img_path1, img_path2,score = str(line.strip()).split(',')
        score = float(score)
        # if score<0.60158:
        if score<0.55:
            print(img_path1[-8:], img_path2[-8:],score)
            if img_path1 not in index_cluster.keys():
                index_cluster[img_path1]=cluster
                cluster+=1
            if img_path2 not in index_cluster.keys():
                index_cluster[img_path2]=cluster
                cluster+=1
        if score>0.78358:
            if img_path1 not in index_cluster.keys() and img_path2 not in index_cluster.keys():
                index_cluster[img_path1]=index_cluster[img_path2]=cluster
                cluster+=1
            elif img_path1 in index_cluster.keys():
                index_cluster[img_path2]=index_cluster[img_path1]
            else:
                index_cluster[img_path1]=index_cluster[img_path2]

with open("./validation2.json", 'w') as f:
    json_dict = json.dumps(index_cluster)
    f.write(json_dict)
    