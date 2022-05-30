# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F

from .utils import euclidean_dist, cosine_dist


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


def hard_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pair wise distance between samples, shape [N, M]
      is_pos: positive index with shape [N, M]
      is_neg: negative index with shape [N, M]
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N]
    dist_ap, _ = torch.max(dist_mat * is_pos, dim=1)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N]
    dist_an, _ = torch.min(dist_mat * is_neg + is_pos * 1e9, dim=1)

    return dist_ap, dist_an


def weighted_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the weighted positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      is_pos:
      is_neg:
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    """
    assert len(dist_mat.size()) == 2

    is_pos = is_pos
    is_neg = is_neg
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg

    weights_ap = softmax_weights(dist_ap, is_pos)
    weights_an = softmax_weights(-dist_an, is_neg)

    dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
    dist_an = torch.sum(dist_an * weights_an, dim=1)

    return dist_ap, dist_an


def triplet_loss(embedding, targets, margin, norm_feat, hard_mining):
    r"""Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    if norm_feat:
        dist_mat = cosine_dist(embedding, embedding)
    else:
        dist_mat = euclidean_dist(embedding, embedding)

    # For distributed training, gather all features from different process.
    # if comm.get_world_size() > 1:
    #     all_embedding = torch.cat(GatherLayer.apply(embedding), dim=0)
    #     all_targets = concat_all_gather(targets)
    # else:
    #     all_embedding = embedding
    #     all_targets = targets

    N = dist_mat.size(0)
    is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
    is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

    if hard_mining:
        dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)
    else:
        dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)

    y = dist_an.new().resize_as_(dist_an).fill_(1)

    if margin > 0:
        loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
    else:
        loss = F.soft_margin_loss(dist_an - dist_ap, y)
        # fmt: off
        if loss == float('Inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
        # fmt: on

    return loss



def triplet_loss_MB(embedding, targets, embeddings_instance, labels_instance, margin=0, norm_feat=True, hard_mining=True):
    r"""Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    if norm_feat:
        dist_mat = cosine_dist(embedding, embeddings_instance)
    else:
        dist_mat = euclidean_dist(embedding, embeddings_instance)

    # For distributed training, gather all features from different process.
    # if comm.get_world_size() > 1:
    #     all_embedding = torch.cat(GatherLayer.apply(embedding), dim=0)
    #     all_targets = concat_all_gather(targets)
    # else:
    #     all_embedding = embedding
    #     all_targets = targets

    N,M = dist_mat.shape #size(0)
    is_pos = targets.view(N, 1).expand(N, M).eq(labels_instance.view(1, M).expand(N, M)).float()
    is_neg = targets.view(N, 1).expand(N, M).ne(labels_instance.view(1, M).expand(N, M)).float()

    # is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
    # is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()
    if hard_mining:
        dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)
    else:
        dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)

    y = dist_an.new().resize_as_(dist_an).fill_(1)

    if margin > 0:
        loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
    else:
        loss = F.soft_margin_loss(dist_an - dist_ap, y)
        # fmt: off
        if loss == float('Inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
        # fmt: on

    return loss


# class ContrastiveLoss(nn.Module):
#     def __init__(self, ):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = 0.5

def contrastiveLoss(inputs_col, targets_col, inputs_row, target_row):
    margin = 0.5
    n = inputs_col.size(0)
    # Compute similarity matrix
    sim_mat = torch.matmul(inputs_col, inputs_row.t())
    epsilon = 1e-5
    loss = list()
    neg_count = list()
    for i in range(n):
        pos_pair_ = torch.masked_select(sim_mat[i], targets_col[i] == target_row)
        pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1 - epsilon)
        neg_pair_ = torch.masked_select(sim_mat[i], targets_col[i] != target_row)

        neg_pair = torch.masked_select(neg_pair_, neg_pair_ > margin)

        pos_loss = torch.sum(-pos_pair_ + 1)
        if len(neg_pair) > 0:
            neg_loss = torch.sum(neg_pair)
            neg_count.append(len(neg_pair))
        else:
            neg_loss = 0

        loss.append(pos_loss + neg_loss)
    if inputs_col.shape[0] == inputs_row.shape[0]:
        prefix = "batch_"
    else:
        prefix = "memory_"
    # if len(neg_count) != 0:
    #     log_info[f"{prefix}average_neg"] = sum(neg_count) / len(neg_count)
    # else:
    #     log_info[f"{prefix}average_neg"] = 0
    # log_info[f"{prefix}non_zero"] = len(neg_count)
    loss = sum(loss) / n  # / all_targets.shape[1]
    return loss