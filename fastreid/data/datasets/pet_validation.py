# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import json
import os.path as osp
import re
import warnings

from cv2 import randn
import random
from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
from collections import defaultdict

@DATASET_REGISTRY.register()
class Pet_Validation(ImageDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = ''
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'
    dataset_name = "pet"

    def __init__(self, root='datasets',  **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'pet_biometric_challenge_2022')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"Market-1501-v15.09.15".')

        self.train_dir = osp.join(self.data_dir, 'train/images')
        self.query_dir = osp.join(self.data_dir, 'train/images')
        self.gallery_dir = osp.join(self.data_dir, 'train/images')
        self.data_label = osp.join(self.data_dir, 'train', 'train_data.csv')
        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)
        train,query,gallery = self.process_dir(self.train_dir)
        # query = self.process_dir(self.query_dir, is_train=False)
        # gallery = self.process_dir(self.gallery_dir, is_train=False)
        super(Pet_Validation, self).__init__(train, query, gallery, **kwargs)
    def process_dir(self, dir_path, is_train=True):
        train = []
        query = []
        gallery = []
        test = defaultdict(list)
        with open(self.data_label, "r") as f:
            lines = f.readlines()
            lines = lines[1:]
            for line in lines:
                # print(str(line.strip()).split(','))
                pid, img_path = str(line.strip()).split(',')
                # if int(pid)<5000:
                # if (int(pid)+1)%10==0:
                #     test[int(pid)].append(img_path)
                # else:
                #     train.append((osp.join(dir_path,img_path), self.dataset_name + "_" + str(pid), self.dataset_name + "_" + str(-1)))
                
                train.append((osp.join(dir_path,img_path), self.dataset_name + "_" + str(pid), self.dataset_name + "_" + str(-1)))
                if (int(pid)+1)%10==0:
                    test[int(pid)].append(img_path)

                # if int(pid)%20==0:

                # if (int(pid)+1)%5==0:
                # # if random.random()<0.15:
                #     train.append((osp.join(dir_path,img_path), self.dataset_name + "_" + str(pid), self.dataset_name + "_" + str(-1)))
                #     test[int(pid)].append(img_path)

            data  = open("./validation2.json",'r')
            data_validataion = json.load(data)
            labels = set()
            for key in data_validataion:
                # labels.add(data_validataion[key])
                lable = data_validataion[key]
                train.append((osp.join("./datasets/pet_biometric_challenge_2022/validation/images/",key), self.dataset_name + "_" + str(lable+6000), self.dataset_name + "_" + str(-1)))
                    # query.append((img_path, int(id), -1))
            print(len(labels))
        for key in test.keys():
            query.append(((osp.join(dir_path,test[key][0]), int(key), -1)))
            for path in test[key][1:]:
                gallery.append(((osp.join(dir_path,path), int(key), -2)))
        return train, query, gallery
        # img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        # pattern = re.compile(r'([-\d]+)_c(\d)')
        # data = []
        # for img_path in img_paths:
        #     pid, camid = map(int, pattern.search(img_path).groups())
        #     if pid == -1:
        #         continue  # junk images are just ignored
        #     assert 0 <= pid <= 1501  # pid == 0 means background
        #     assert 1 <= camid <= 6
        #     camid -= 1  # index starts from 0
        #     if is_train:
        #         pid = self.dataset_name + "_" + str(pid)
        #         camid = self.dataset_name + "_" + str(camid)
        #     data.append((img_path, pid, camid))
        # return data
