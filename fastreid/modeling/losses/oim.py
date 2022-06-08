import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
import math
import random
from .triplet_loss import *


def TripletLossOim(inputs_col, targets_col, inputs_row, targets_row):
    margin = 0.1
    n = inputs_col.size(0)
    # Compute similarity matrix
    sim_mat = torch.matmul(inputs_col, inputs_row.t())
    # split the positive and negative pairs
    eyes_ = torch.eye(n, dtype=torch.uint8).cuda()
    pos_mask = targets_col.expand(
        targets_row.shape[0], n
    ).t() == targets_row.expand(n, targets_row.shape[0])
    neg_mask = ~ pos_mask #1 - pos_mask
    # pos_mask[:, :n] = pos_mask[:, :n] * (~eyes_) #pos_mask[:, :n] - eyes_
    loss = list()
    neg_count = list()
    for i in range(n):
        pos_pair_idx = torch.nonzero(pos_mask[i, :]).view(-1)
        if pos_pair_idx.shape[0] > 0:
            pos_pair_ = sim_mat[i, pos_pair_idx]
            pos_pair_ = torch.sort(pos_pair_)[0]

            neg_pair_idx = torch.nonzero(neg_mask[i, :]).view(-1)
            neg_pair_ = sim_mat[i, neg_pair_idx]
            neg_pair_ = torch.sort(neg_pair_)[0]

            select_pos_pair_idx = torch.nonzero(
                pos_pair_ < neg_pair_[-1] + margin
            ).view(-1)
            pos_pair = pos_pair_[select_pos_pair_idx]

            select_neg_pair_idx = torch.nonzero(
                neg_pair_ > max(0.6, pos_pair_[-1]) - margin
            ).view(-1)
            neg_pair = neg_pair_[select_neg_pair_idx]

            pos_loss = torch.sum(1 - pos_pair)
            if len(neg_pair) >= 1:
                neg_loss = torch.sum(neg_pair)
                neg_count.append(len(neg_pair))
            else:
                neg_loss = 0
            loss.append(pos_loss + neg_loss)
        else:
            loss.append(0)
    loss = sum(loss) / n

    return loss

# class MultiSimilarityLoss(nn.Module):
#     def __init__(self, ):
#         super(MultiSimilarityLoss, self).__init__()
def MultiSimilarityLoss(inputs_col, targets_col, inputs_row, target_row):
    thresh = 0.5
    margin = 0.1
    scale_pos = 2.0 #cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_POS
    scale_neg = 40.0 #cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_NEG
    hard_mining = True #cfg.LOSSES.MULTI_SIMILARITY_LOSS.HARD_MINING
    batch_size = inputs_col.size(0)
    sim_mat = torch.matmul(inputs_col, inputs_row.t())

    epsilon = 1e-5
    loss = list()
    neg_count = 0
    for i in range(batch_size):
        pos_pair_ = torch.masked_select(sim_mat[i], target_row == targets_col[i])
        pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1 - epsilon)
        neg_pair_ = torch.masked_select(sim_mat[i], target_row != targets_col[i])
        # sampling step
        if hard_mining:
            neg_pair = neg_pair_[neg_pair_ + margin > torch.min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - margin < torch.max(neg_pair_)]

        if len(neg_pair) < 1 or len(pos_pair) < 1:
            continue
        neg_count += len(neg_pair)

        # weighting step
        pos_loss = (
            1.0
            / scale_pos
            * torch.log(
                1 + torch.sum(torch.exp(-scale_pos * (pos_pair - thresh)))
            )
        )
        neg_loss = (
            1.0
            / scale_neg
            * torch.log(
                1 + torch.sum(torch.exp(scale_neg * (neg_pair - thresh)))
            )
        )
        loss.append(pos_loss + neg_loss)

    if len(loss) == 0:
        return torch.zeros([], requires_grad=True).cuda()
    # log_info["neg_count"] = neg_count / batch_size
    loss = sum(loss) / batch_size
    return loss


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
    	num_classes (int): number of classes.
    	epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, reduce=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.reduction = reduce
    def forward(self, inputs, targets):
        """
        Args:
        	inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        	targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class OIM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())
        return outputs
    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()
        return grad_inputs, None, None, None
def oim(inputs, indexes, features, momentum=0.5):
    return OIM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))
class OIMLoss_ori(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.ce = CrossEntropyLabelSmooth(num_samples)
        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets):
        inputs = F.normalize(inputs, dim=1).cuda()
        outputs = oim(inputs, targets, self.features, self.momentum)
        print(outputs)
        outputs /= self.temp
        loss = F.cross_entropy(outputs, targets)
        # loss = self.ce(outputs, targets)
        return loss
        
class OIMLoss(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.ce = CrossEntropyLabelSmooth(num_samples)
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('sample_features', torch.zeros(num_samples, num_features))
        self.register_buffer('sample_labels', torch.zeros(num_samples,))
    def forward(self, inputs, targets):
        inputs = F.normalize(inputs, dim=1).cuda()
        outputs = oim(inputs, targets, self.features, self.momentum)
        outputs2 = inputs.mm(self.sample_features.t())
        temp_sims = outputs2.detach().clone()
        # print(outputs)
        outputs /= self.temp
        outputs2 /= self.temp
        # loss = F.cross_entropy(outputs, targets)
        # loss2 = MultiSimilarityLoss(inputs,targets,self.sample_features,self.sample_labels)
        # loss2 = F.cross_entropy(outputs2, targets)
        # loss2 = triplet_loss_MB(inputs,targets,self.sample_features,self.sample_labels)
        # loss2 = triplet_loss_MB(inputs,targets,self.sample_features,self.sample_labels)
        # loss2 = contrastiveLoss(inputs,targets,self.sample_features,self.sample_labels)
        # loss2 = contrastiveLoss(inputs,targets,self.sample_features,self.sample_labels)
        # loss2 = TripletLossOim(inputs,targets,self.sample_features,self.sample_labels)
        # associate_loss=0
        # for k in range(len(targets)):
        #     ori_asso_ind = torch.nonzero(self.sample_labels == targets[k]).squeeze(-1)
        #     # weights = F.softmax(1-temp_sims[k, ori_asso_ind],dim=0)
        #     temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
        #     sel_ind = torch.sort(temp_sims[k])[1][-500:]
        #     min_index = torch.min(outputs2[k, ori_asso_ind],dim=0)[1]
        #     # import pdb; pdb.set_trace()
        #     # print(outputs2[k, ori_asso_ind][[min_index,]])
        #     concated_input = torch.cat((outputs2[k, ori_asso_ind][[int(min_index),]], outputs2[k, sel_ind]), dim=0)
        #     concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(torch.device('cuda'))
        #     concated_target[0] = 1.0 #weights #1.0 / len(ori_asso_ind) :len(ori_asso_ind)
        #     associate_loss += -1 * (F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
        # loss2 = 0.5*associate_loss / len(targets)
        associate_loss=0
        for k in range(len(targets)):
            ori_asso_ind = torch.nonzero(self.sample_labels == targets[k]).squeeze(-1)
            weights = F.softmax(1-temp_sims[k, ori_asso_ind],dim=0)
            temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
            sel_ind = torch.sort(temp_sims[k])[1][-500:]
            concated_input = torch.cat((outputs2[k, ori_asso_ind], outputs2[k, sel_ind]), dim=0)
            concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(torch.device('cuda'))
            concated_target[0:len(ori_asso_ind)] = weights #1.0 / len(ori_asso_ind)
            associate_loss += -1 * (F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
        loss_wincetance = 0.5 * associate_loss / len(targets)
        # loss +=loss2
        loss = self.ce(outputs, targets)
        # return loss, loss2
        return loss, loss_wincetance


class OIMLoss_tri(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(OIMLoss_tri, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.ce = CrossEntropyLabelSmooth(num_samples)
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('sample_features', torch.zeros(num_samples, num_features))
        self.register_buffer('sample_labels', torch.zeros(num_samples,))
    def forward(self, inputs, targets):
        inputs = F.normalize(inputs, dim=1).cuda()
        outputs = oim(inputs, targets, self.features, self.momentum)
        outputs2 = inputs.mm(self.sample_features.t())
        temp_sims = outputs2.detach().clone()
        # print(outputs)
        outputs /= self.temp
        outputs2 /= self.temp
        # loss = F.cross_entropy(outputs, targets)
        # loss2 = MultiSimilarityLoss(inputs,targets,self.sample_features,self.sample_labels)
        # loss2 = F.cross_entropy(outputs2, targets)
        # loss2 = triplet_loss_MB(inputs,targets,self.sample_features,self.sample_labels)
        # loss2 = triplet_loss_MB(inputs,targets,self.sample_features,self.sample_labels)
        # loss2 = contrastiveLoss(inputs,targets,self.sample_features,self.sample_labels)
        # loss2 = contrastiveLoss(inputs,targets,self.sample_features,self.sample_labels)
        loss2 = TripletLossOim(inputs,targets,self.sample_features,self.sample_labels)
        loss = self.ce(outputs, targets)
        # return loss, loss2
        return loss, loss2

class OIMLoss_con(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(OIMLoss_con, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.ce = CrossEntropyLabelSmooth(num_samples)
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('sample_features', torch.zeros(num_samples, num_features))
        self.register_buffer('sample_labels', torch.zeros(num_samples,))
    def forward(self, inputs, targets):
        inputs = F.normalize(inputs, dim=1).cuda()
        outputs = oim(inputs, targets, self.features, self.momentum)
        outputs2 = inputs.mm(self.sample_features.t())
        # temp_sims = outputs2.detach().clone()
        # print(outputs)
        outputs /= self.temp
        outputs2 /= self.temp
        # loss = F.cross_entropy(outputs, targets)
        # loss2 = MultiSimilarityLoss(inputs,targets,self.sample_features,self.sample_labels)
        # loss2 = F.cross_entropy(outputs2, targets)
        # loss2 = triplet_loss_MB(inputs,targets,self.sample_features,self.sample_labels)
        # loss2 = triplet_loss_MB(inputs,targets,self.sample_features,self.sample_labels)
        # loss2 = contrastiveLoss(inputs,targets,self.sample_features,self.sample_labels)
        loss2 = contrastiveLoss(inputs,targets,self.sample_features,self.sample_labels)
        # loss2 = TripletLossOim(inputs,targets,self.sample_features,self.sample_labels)
        loss = self.ce(outputs, targets)
        # return loss, loss2
        return loss, loss2

class OIMLoss_MS(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(OIMLoss_MS, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.ce = CrossEntropyLabelSmooth(num_samples)
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('sample_features', torch.zeros(num_samples, num_features))
        self.register_buffer('sample_labels', torch.zeros(num_samples,))
    def forward(self, inputs, targets):
        inputs = F.normalize(inputs, dim=1).cuda()
        outputs = oim(inputs, targets, self.features, self.momentum)
        outputs2 = inputs.mm(self.sample_features.t())
        temp_sims = outputs2.detach().clone()
        # print(outputs)
        outputs /= self.temp
        outputs2 /= self.temp
        # loss = F.cross_entropy(outputs, targets)
        loss2 = MultiSimilarityLoss(inputs,targets,self.sample_features,self.sample_labels)
        # loss2 = F.cross_entropy(outputs2, targets)
        # loss2 = triplet_loss_MB(inputs,targets,self.sample_features,self.sample_labels)
        # loss2 = triplet_loss_MB(inputs,targets,self.sample_features,self.sample_labels)
        # loss2 = contrastiveLoss(inputs,targets,self.sample_features,self.sample_labels)
        # loss2 = contrastiveLoss(inputs,targets,self.sample_features,self.sample_labels)
        # loss2 = TripletLossOim(inputs,targets,self.sample_features,self.sample_labels)
        loss = self.ce(outputs, targets)
        # return loss, loss2
        return loss, loss2

class OIMLoss_Mininstance(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(OIMLoss_Mininstance, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.ce = CrossEntropyLabelSmooth(num_samples)
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('sample_features', torch.zeros(num_samples, num_features))
        self.register_buffer('sample_labels', torch.zeros(num_samples,))
    def forward(self, inputs, targets):
        inputs = F.normalize(inputs, dim=1).cuda()
        outputs = oim(inputs, targets, self.features, self.momentum)
        outputs2 = inputs.mm(self.sample_features.t())
        temp_sims = outputs2.detach().clone()
        # print(outputs)
        outputs /= self.temp
        outputs2 /= self.temp
        # loss = F.cross_entropy(outputs, targets)
        # loss2 = MultiSimilarityLoss(inputs,targets,self.sample_features,self.sample_labels)
        # loss2 = F.cross_entropy(outputs2, targets)
        # loss2 = triplet_loss_MB(inputs,targets,self.sample_features,self.sample_labels)
        # loss2 = triplet_loss_MB(inputs,targets,self.sample_features,self.sample_labels)
        # loss2 = contrastiveLoss(inputs,targets,self.sample_features,self.sample_labels)
        # loss2 = contrastiveLoss(inputs,targets,self.sample_features,self.sample_labels)
        # loss2 = TripletLossOim(inputs,targets,self.sample_features,self.sample_labels)
        associate_loss=0
        for k in range(len(targets)):
            ori_asso_ind = torch.nonzero(self.sample_labels == targets[k]).squeeze(-1)
            # weights = F.softmax(1-temp_sims[k, ori_asso_ind],dim=0)
            temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
            sel_ind = torch.sort(temp_sims[k])[1][-500:]
            min_index = torch.min(outputs2[k, ori_asso_ind],dim=0)[1]
            # import pdb; pdb.set_trace()
            # print(outputs2[k, ori_asso_ind][[min_index,]])
            concated_input = torch.cat((outputs2[k, ori_asso_ind][[int(min_index),]], outputs2[k, sel_ind]), dim=0)
            concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(torch.device('cuda'))
            concated_target[0] = 1.0 #weights #1.0 / len(ori_asso_ind) :len(ori_asso_ind)
            associate_loss += -1 * (F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
        loss_mincetance = 0.5*associate_loss / len(targets)

        # associate_loss=0
        # for k in range(len(targets)):
        #     ori_asso_ind = torch.nonzero(self.sample_labels == targets[k]).squeeze(-1)
        #     weights = F.softmax(1-temp_sims[k, ori_asso_ind],dim=0)
        #     temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
        #     sel_ind = torch.sort(temp_sims[k])[1][-500:]
        #     concated_input = torch.cat((outputs2[k, ori_asso_ind], outputs2[k, sel_ind]), dim=0)
        #     concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(torch.device('cuda'))
        #     concated_target[0:len(ori_asso_ind)] = weights #1.0 / len(ori_asso_ind)
        #     associate_loss += -1 * (F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
        # loss_mincetance = 0.5 * associate_loss / len(targets)
        # # loss +=loss2
        loss = self.ce(outputs, targets)
        # return loss, loss2
        return loss, loss_mincetance

class OIMLoss_siameseoffline(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(OIMLoss_siameseoffline, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.ce = CrossEntropyLabelSmooth(num_samples)
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('sample_features', torch.zeros(num_samples, num_features))
        self.register_buffer('sample_labels', torch.zeros(num_samples,))
        self.fully_connect1 = torch.nn.Linear(2048, 512)
        self.fully_connect2 = torch.nn.Linear(512, 1)
        self.loss_mse = nn.BCEWithLogitsLoss()
        self.relu = nn.LeakyReLU()
    def forward(self, inputs, targets=None):
        inputs = F.normalize(inputs, dim=1).cuda()
        if self.training:
        # outputs = oim(inputs, targets, self.features, self.momentum)
            outputs2 = inputs.mm(self.sample_features.t())
            temp_sims = outputs2.detach().clone()
            # print(outputs)
            # outputs /= self.temp
            outputs2 /= self.temp
            # loss = F.cross_entropy(outputs, targets)
            # loss2 = MultiSimilarityLoss(inputs,targets,self.sample_features,self.sample_labels)
            # loss2 = F.cross_entropy(outputs2, targets)
            # loss2 = triplet_loss_MB(inputs,targets,self.sample_features,self.sample_labels)
            # loss2 = triplet_loss_MB(inputs,targets,self.sample_features,self.sample_labels)
            # loss2 = contrastiveLoss(inputs,targets,self.sample_features,self.sample_labels)
            # loss2 = contrastiveLoss(inputs,targets,self.sample_features,self.sample_labels)
            # loss2 = TripletLossOim(inputs,targets,self.sample_features,self.sample_labels)
            # associate_loss=0
            # for k in range(len(targets)):
            #     ori_asso_ind = torch.nonzero(self.sample_labels == targets[k]).squeeze(-1)
            #     # weights = F.softmax(1-temp_sims[k, ori_asso_ind],dim=0)
            #     temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
            #     sel_ind = torch.sort(temp_sims[k])[1][-500:]
            #     min_index = torch.min(outputs2[k, ori_asso_ind],dim=0)[1]
            #     # import pdb; pdb.set_trace()
            #     # print(outputs2[k, ori_asso_ind][[min_index,]])
            #     concated_input = torch.cat((outputs2[k, ori_asso_ind][[int(min_index),]], outputs2[k, sel_ind]), dim=0)
            #     concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(torch.device('cuda'))
            #     concated_target[0] = 1.0 #weights #1.0 / len(ori_asso_ind) :len(ori_asso_ind)
            #     associate_loss += -1 * (F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
            # loss2 = 0.5*associate_loss / len(targets)
            x2_neg = []
            x2_pos = []
            associate_loss=0
            for k in range(len(targets)):
                ori_asso_ind = torch.nonzero(self.sample_labels == targets[k]).squeeze(-1)
                # weights = F.softmax(1-temp_sims[k, ori_asso_ind],dim=0)
                temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
                sel_ind = torch.sort(temp_sims[k])[1][-500:]
                # concated_input = torch.cat((outputs2[k, ori_asso_ind], outputs2[k, sel_ind]), dim=0)
                # concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(torch.device('cuda'))
                # concated_target[0:len(ori_asso_ind)] = weights #1.0 / len(ori_asso_ind)
                # associate_loss += -1 * (F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
                # positive_features = self.features[ori_asso_ind]
                # negtive_features = self.features[~ori_asso_ind]
                x2_neg.append(self.sample_features[random.choice(sel_ind)])   #hard negtive 
                x2_pos.append(self.sample_features[random.choice(ori_asso_ind)]) #positive loss
            x2_neg = torch.stack(x2_neg)
            x2_pos = torch.stack(x2_pos)
            x_neg =torch.abs(inputs-x2_neg)
            x_neg = self.relu(self.fully_connect1(x_neg))
            x_neg = self.fully_connect2(x_neg)
            x_pos = torch.abs(inputs-x2_pos)
            x_pos = self.relu(self.fully_connect1(x_pos))
            x_pos = self.fully_connect2(x_pos)
            labels_negtive = torch.zeros((len(targets),1)).cuda()
            labels_positive = torch.ones((len(targets), 1)).cuda()
            # return x
            # loss2 = 0.5 * associate_loss / len(targets)
            # loss +=loss2
            # loss = self.ce(outputs, targets)
            # return loss, loss2
            loss_positive = self.loss_mse(x_pos, labels_positive)
            loss_negtive = self.loss_mse(x_neg, labels_negtive)
            loss_siamese = loss_positive+loss_negtive
            return 0, loss_siamese
        else:
            x1= inputs[0]
            x2= inputs[1]
            predicts =torch.abs(x1-x2)
            predicts = self.relu(self.fully_connect1(predicts))
            predicts = self.fully_connect2(predicts)
            predicts = nn.Sigmoid()(predicts)
            return predicts


class OIMLoss2_oriinstance(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(OIMLoss2_oriinstance, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.ce = CrossEntropyLabelSmooth(num_samples)
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('sample_features', torch.zeros(num_samples, num_features))
        self.register_buffer('sample_labels', torch.zeros(num_samples,))

    def forward(self, inputs, targets):
        inputs = F.normalize(inputs, dim=1).cuda()
        outputs = oim(inputs, targets, self.features, self.momentum)
        outputs2 = inputs.mm(self.sample_features.t())
        temp_sims = outputs2.detach().clone()
        # print(outputs)
        outputs /= self.temp
        outputs2 /= self.temp
        loss = F.cross_entropy(outputs, targets)
        # loss2 = F.cross_entropy(outputs2, targets)
        associate_loss=0
        for k in range(len(targets)):
            ori_asso_ind = torch.nonzero(self.sample_labels == targets[k]).squeeze(-1)
            temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
            sel_ind = torch.sort(temp_sims[k])[1][-500:]
            concated_input = torch.cat((outputs2[k, ori_asso_ind], outputs2[k, sel_ind]), dim=0)
            concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(torch.device('cuda'))
            concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
            associate_loss += -1 * (F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
        loss_instance = 0.5 * associate_loss / len(targets)
        # loss +=loss2
        # loss = self.ce(outputs, targets)
        return loss, loss_instance     

class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()
class OIMLossArc(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(OIMLossArc, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.ce = CrossEntropyLabelSmooth(num_samples)
        self.cri = DenseCrossEntropy()
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        # self.regis
        self.s = 30
        self.m = 0.3 #0.5
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, inputs, targets):
        # print(inputs.shape)
        inputs = F.normalize(inputs, dim=1).cuda()
        cosine = oim(inputs, targets, self.features, self.momentum)
        # print(outputs)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)
        # outputs /= self.temp
        loss = F.cross_entropy(output, targets)
        # loss = self.cri(output, targets)
        # loss = self.ce(output, targets)
        return loss


class OIMLosswithTri(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(OIMLosswithTri, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.ce = CrossEntropyLabelSmooth(num_samples)
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('sample_features', torch.zeros(num_samples, num_features))
        self.register_buffer('sample_labels', torch.zeros(num_samples,))

    def forward(self, inputs, targets):
        inputs = F.normalize(inputs, dim=1).cuda()
        outputs = oim(inputs, targets, self.features, self.momentum)
        outputs2 = inputs.mm(self.sample_features.t())
        temp_sims = outputs2.detach().clone()
        # print(outputs)
        outputs /= self.temp
        outputs2 /= self.temp
        loss = F.cross_entropy(outputs, targets)
        # loss2 = F.cross_entropy(outputs2, targets)
        associate_loss=0
        for k in range(len(targets)):
            ori_asso_ind = torch.nonzero(self.sample_labels == targets[k]).squeeze(-1)
            temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
            sel_ind = torch.sort(temp_sims[k])[1][-500:]
            concated_input = torch.cat((outputs2[k, ori_asso_ind], outputs2[k, sel_ind]), dim=0)
            concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(torch.device('cuda'))
            concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
            associate_loss += -1 * (F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
        loss2 = 0.5 * associate_loss / len(targets)
        # loss +=loss2
        # loss = self.ce(outputs, targets)
        return loss, loss2

loss_dict = {
"OIMLoss_ori":OIMLoss_ori,
"OIMLoss":OIMLoss,
"OIMLoss_Mininstance":OIMLoss_Mininstance,
"OIMLoss_siameseoffline":OIMLoss_siameseoffline,
"OIMLoss2_oriinstance":OIMLoss2_oriinstance,
"OIMLossArc":OIMLossArc,
"OIMLoss_tri":OIMLoss_tri,
"OIMLoss_con":OIMLoss_con,
"OIMLoss_MS":OIMLoss_MS
}
