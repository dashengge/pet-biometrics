from __future__ import absolute_import

import torch
from torch import nn
import numpy as np
from torch.distributions import Normal
import torch.nn.functional as F


class VarianceLoss(nn.Module):
    def __init__(self, loss2_wight=1, loss3_wight=0.1):
        super(VarianceLoss, self).__init__()

        self.cross_loss = nn.CrossEntropyLoss()
        self.loss2_wight = loss2_wight
        self.loss3_wight = loss3_wight

    def forward(self, mu, sig, logits2, label):
        # assert mu.size(0) == label.size(0), "features.size(0) is not equal to labels.size(0)"
        sigma_avg = 4
        threshold = np.log(sigma_avg) + (1 + np.log(2 * np.pi)) / 2
        losses = torch.mean(torch.nn.functional.relu(threshold - Normal(loc=mu, scale=sig).entropy()))
        num = len(logits2)
        if num != 0:
            softloss2 = self.cross_loss(logits2[0], label)
            for logit_ in logits2[1:]:
                softloss2 = softloss2 + self.cross_loss(logit_, label)
            total_loss = self.loss2_wight * softloss2 / num + self.loss3_wight * losses
        else:
            total_loss = self.loss3_wight * losses
        return total_loss
