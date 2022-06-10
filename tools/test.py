#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from collections import defaultdict
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

def batch_torch_topk_self(qf, gf, k1, N=1000, query = False):
    # m = qf.shape[0]
    # n = gf.shape[0]
    m = gf.shape[0]
    n = qf.shape[0]
    dist_mat = []
    initial_rank = []
    for j in tqdm(range(n // N + 1)):
        # temp_gf = gf[j * N:j * N + N]
        temp_qf = qf[j * N:j * N + N]
        if len(temp_qf)==0:
            continue
        temp_qd = []
        for i in range(m // N + 1):
            # temp_qf = qf[i * N:i * N + N]
            temp_gf = gf[i * N:i * N + N]
            temp_d = torch.mm(temp_gf, temp_qf.t())
            temp_qd.append(temp_d)
        temp_qd = torch.cat(temp_qd, dim=0)
        temp_qd = temp_qd / (torch.max(temp_qd, dim=0)[0])
        temp_qd = temp_qd.t()
        value, rank = torch.topk(temp_qd, k=k1, dim=1, largest=True, sorted=True)
        # rank = torch.topk(temp_qd, k=k1, dim=1, largest=False, sorted=True)[1]
        if query:
            dist_mat.append(value)
        initial_rank.append(rank)
    del value
    del rank
    del temp_qd
    del temp_gf
    del temp_qf
    del temp_d
    torch.cuda.empty_cache()  # empty GPU memory

    initial_rank = torch.cat(initial_rank, dim=0)#.cpu().numpy()
    if query:
        dist_mat = torch.cat(dist_mat, dim=0)  #.cpu().numpy()
        return dist_mat, initial_rank
    return initial_rank

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    flip_test=True
    cfg = setup(args)
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    model = build_model(cfg)
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
    model.cuda()
    transforms = build_transforms(cfg, is_train=False)
    # print(model(x).shape)
    data_dir = "./datasets/pet_biometric_challenge_2022/test/test"
    with open("./datasets/pet_biometric_challenge_2022/test/test_data.csv", "r") as f:
        data =f.readlines()
        data = data[1:]
    inputs={}
    inputs["targets"] = torch.zeros([2,]).cuda()

    # all_img = []
    # for line in tqdm(data):
    #     img_path1, img_path2 = str(line.strip()).split(',')
    #     all_img.append(img_path1)
    #     all_img.append(img_path2)
    # all_img = list(set(all_img))
    # img_tensors = {}
    # all_embeddings = []
    # for img_path in tqdm(all_img):
    #     img = read_image(os.path.join(data_dir, img_path))
    #     img = transforms(img)
    #     img = img.cuda()
    #     img = torch.unsqueeze(img, dim=0)
    #     out_embeddings = model(img)
    #     out_embeddings = out_embeddings.detach().cpu()
    #     img_tensors[img_path] = out_embeddings
    # keys = img_tensors.keys()
    # for key in keys:
    #     all_embeddings.append(img_tensors[key])
    # X_g = torch.cat(all_embeddings)
    # print(X_g.shape)
    # S, initial_rank = batch_torch_topk_self(X_g, X_g, k1=3, query=True)
    # img_tensors_enhance = {}
    # for i, key in enumerate(keys):
    #     features = X_g[initial_rank[i]]
    #     # features = torch.cat([X_g[i].unsqueeze(dim=0), features], dim=0)
    #     weight = torch.logspace(0, -2, 3) #.cuda()
    #     features = features* weight.unsqueeze(-1)
    #     features = torch.sum(features,dim=0)
    #     print(features.shape)
    #     img_tensors_enhance[key]=features

    # with open("/home/lzy/fast-reid/datasets/pet_biometric_challenge_2022/validation/submit_all_data_rerank.csv", "w") as f1:
    #     for line in tqdm(data):
    #         img_path1, img_path2 = str(line.strip()).split(',')
    #         tensor1 = img_tensors_enhance[img_path1]
    #         tensor2 = img_tensors_enhance[img_path2] 
    #         similarity = torch.cosine_similarity(torch.unsqueeze(tensor1,dim=0),torch.unsqueeze(tensor2,dim=0)).item()
    #         f1.write("{},{},{}\n".format(img_path1,img_path2,(similarity+1)/2))

    # with open("./fast_resnetst101_tta.csv", "w") as f1:
    #     f1.write("imageA,imageB,prediction\n")
    #     for line in tqdm(data):
    #         img_path1, img_path2 = str(line.strip()).split(',')
    #         img1 = read_image(os.path.join(data_dir, img_path1))
    #         img2 = read_image(os.path.join(data_dir, img_path2))
    #         img1 = transforms(img1)
    #         img2 = transforms(img2)
    #         input_img = torch.stack([img1, img2])
    #         input_img = input_img.cuda()
    #         out_embeddings = torch.FloatTensor(input_img.size(0), 2048).zero_().cuda()
    #         for i in range(2):
    #             if i == 1:
    #                 inv_idx = torch.arange(input_img.size(3) - 1, -1, -1).long().cuda()
    #                 input_img = input_img.index_select(3, inv_idx)
    #             f = model(input_img)
    #             out_embeddings = out_embeddings + f
    #         out_embeddings = out_embeddings/2
    #         # out_embeddings = model(input_img)
    #         similarity = torch.cosine_similarity(torch.unsqueeze(out_embeddings[0],dim=0),torch.unsqueeze(out_embeddings[1],dim=0)).item()
    #         f1.write("{},{},{}\n".format(img_path1,img_path2,(similarity+1)/2))

    for line in tqdm(data):
        img_path1, img_path2 = str(line.strip()).split(',')
        img1 = read_image(os.path.join(data_dir, img_path1))
        img2 = read_image(os.path.join(data_dir, img_path2))
        img1 = transforms(img1)
        img2 = transforms(img2)
        input_img = torch.stack([img1, img2])
        input_img = input_img.cuda()
        # input_img_flip=input_img.clone().flip(dims=[3])
        inputs["images"]=input_img
        out_embeddings = model(inputs, True)
        out_embeddings = out_embeddings["features"] #: feat
            # Flip test
            # if flip_test:
            #     # print(input_img.shape)
            #     # inputs = input_img.flip(dims=[3])
            #     inputs["images"]=input_img_flip
            #     flip_outputs = model(inputs, True)
            #     # out_embeddings = (out_embeddings + flip_outputs) / 2
            #     flip_outputs = flip_outputs["features"] #: feat
            #     out_embeddings = torch.cat([out_embeddings , flip_outputs], dim=1)
            #     # print(out_embeddings.shape)
            #     similarity = torch.cosine_similarity(torch.unsqueeze(out_embeddings[0],dim=0),torch.unsqueeze(out_embeddings[1],dim=0)).item()
            #     similarity = (similarity+1)/2
            # else:
            #     similarity = torch.cosine_similarity(torch.unsqueeze(out_embeddings[0],dim=0),torch.unsqueeze(out_embeddings[1],dim=0)).item()
            #     similarity = (similarity+1)/2
            # f1.write("{},{},{}\n".format(img_path1,img_path2,similarity))
    submit_file = args.submit_file
    with open(submit_file, "w") as f1:
        with inference_context(model), torch.no_grad():
            f1.write("imageA,imageB,prediction\n")
            for line in tqdm(data):
                img_path1, img_path2 = str(line.strip()).split(',')
                img1 = read_image(os.path.join(data_dir, img_path1))
                img2 = read_image(os.path.join(data_dir, img_path2))
                img1 = transforms(img1)
                img2 = transforms(img2)
                input_img = torch.stack([img1, img2])
                input_img = input_img.cuda()
                input_img_flip=input_img.clone().flip(dims=[3])
                out_embeddings = model(input_img)
                # Flip test
                if flip_test:
                    flip_outputs = model(input_img_flip)
                    out_embeddings = torch.cat([out_embeddings , flip_outputs], dim=1)
                    similarity = torch.cosine_similarity(torch.unsqueeze(out_embeddings[0],dim=0),torch.unsqueeze(out_embeddings[1],dim=0)).item()
                    similarity = (similarity+1)/2
                else:
                    similarity = torch.cosine_similarity(torch.unsqueeze(out_embeddings[0],dim=0),torch.unsqueeze(out_embeddings[1],dim=0)).item()
                    similarity = (similarity+1)/2
                f1.write("{},{},{}\n".format(img_path1,img_path2,similarity))

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )