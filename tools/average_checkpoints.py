#!/usr/bin/env python
# encoding: utf-8
import argparse
from collections import defaultdict
from pyexpat import model
import sys
from matplotlib import lines
import os
sys.path.append('.')
import torch
from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
from fastreid.modeling.meta_arch import build_model
from fastreid.data.transforms import build_transforms
from fastreid.data.data_utils import read_image
from tqdm import tqdm
from fastreid.evaluation.evaluator import inference_context
# Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
# model.cuda()
# model.eval()
avg_state_dict = {}
avg_counts = {}

def load_state_dict(path):
    state_dict = torch.load(path, map_location=torch.device("cpu"))
    return state_dict['model']


def default_argument_parser():

    parser = argparse.ArgumentParser(description="average checkpoints")
    parser.add_argument("--log-dir", default="", metavar="FILE", help="path to config file")
    return parser
parser = default_argument_parser().parse_args()
log_dir = parser.log_dir
state_dicts = [
         log_dir+"/model_0064.pth",
         log_dir+"/model_0069.pth",
         log_dir+"/model_0074.pth",
         log_dir+"/model_0079.pth",
         log_dir+"/model_0084.pth",
         log_dir+"/model_final.pth",
         ]
    
# print(state_dict.keys())
# for k, v in state_dict['model'].items():
#     print(k)
avg_state_dict = {}
avg_counts = {}
for c in state_dicts:
    new_state_dict = load_state_dict(c)
    if not new_state_dict:
        print("Error: Checkpoint ({}) doesn't exist".format(c))
        continue
    for k, v in new_state_dict.items():
        if k not in avg_state_dict:
            avg_state_dict[k] = v.clone().to(dtype=torch.float64)
            avg_counts[k] = 1
        else:
            avg_state_dict[k] += v.to(dtype=torch.float64)
            avg_counts[k] += 1

for k, v in avg_state_dict.items():
    v.div_(avg_counts[k])
avg_checkpoint=torch.load(state_dicts[-1], map_location=torch.device("cpu"))
avg_checkpoint["model"]=avg_state_dict
torch.save(avg_checkpoint, log_dir+"/avg_model.pth")