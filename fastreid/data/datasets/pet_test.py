from collections import defaultdict
from email.policy import default
from re import L
import numpy as np
import pandas as pd

# def process_dir(self, dir_path, is_train=True):
#     img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
#     pattern = re.compile(r'([-\d]+)_c(\d)')

#     data = []
#     for img_path in img_paths:
#         pid, camid = map(int, pattern.search(img_path).groups())
#         if pid == -1:
#             continue  # junk images are just ignored
#         assert 0 <= pid <= 1501  # pid == 0 means background
#         assert 1 <= camid <= 6
#         camid -= 1  # index starts from 0
#         if is_train:
#             pid = self.dataset_name + "_" + str(pid)
#             camid = self.dataset_name + "_" + str(camid)
#         data.append((img_path, pid, camid))
#     return data
