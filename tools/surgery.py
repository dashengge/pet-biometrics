from cv2 import sort
import numpy as np
import json

cluster=0
index_cluster={}
with open("./datasets/pet_biometric_challenge_2022/validation/submit.csv", "r") as f1:
    data = f1.readlines()
    data = data[1:]
    img_paths1,img_paths2,scores = [],[],[]
    for line in data:
        img_path1, img_path2,socre = str(line.strip()).split(',')
        score = float(score)
        if score<0.6158:
            if img_path1 not in index_cluster.keys():
                index_cluster[img_path1]=cluster
                cluster+=1
            if img_path2 not in index_cluster.keys():
                index_cluster[img_path2]=cluster
                cluster+=1
with open("./validation.json", 'w') as f:
    json_dict = json.dumps(index_cluster)
    f.write(json_dict)