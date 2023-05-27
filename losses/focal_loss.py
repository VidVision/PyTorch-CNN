#Ref:
# https://arxiv.org/pdf/1901.05555.pdf
# https://towardsdatascience.com/handling-class-imbalanced-data-using-a-loss-specifically-made-for-it-6e58fd65ffab

"""
Focal Loss Wrapper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def reweight(cls_num_list, beta=0.9999):
    """
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    """
    per_cls_weights = None
    #############################################################################
    # TODO: reweight each class by effective numbers                            #
    #############################################################################

    num_classes = len(cls_num_list)
    #  classes: [5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50]
    # print('\n \n classes:',cls_num_list )
    # print('\n \n classes and number in classes:',num_classes )
    alpha = (1.0-beta)/(1.0-np.power(beta, cls_num_list))
    #  weight term should be normalized to sum to C based on the paper
    alpha = alpha / np.sum(alpha) * num_classes
    per_cls_weights = torch.tensor(alpha)

    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = None
        #############################################################################
        # TODO: Implement forward pass of the focal loss                            #
        ############################################################################
        # https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/7
        # https://towardsdatascience.com/handling-class-imbalanced-data-using-a-loss-specifically-made-for-it-6e58fd65ffab
        # CB Focal Loss = - alpha * sum_over classes ((1-pi)**gamma * log(pi))
        # alpha = self.weight.float() # torch.Size([10])
        gamma = self.gamma
        num_classes = self.weight.shape[0]
        one_hot_target =F.one_hot(target, num_classes)  # torch.Size([128, 10])

        # calculate probability (softmax)
        numerator = torch.exp(input)  # torch.Size([128, 10])
        denum = torch.sum(numerator, dim=1).unsqueeze(1).repeat(1, num_classes) # torch.Size([128, 10])
        p = numerator/denum # torch.Size([128, 10])
        # print('\n \n prob, prob shape:', p, p.shape)

        weighted_loss = -self.weight * (1.0-p) ** gamma * torch.log(p)  # torch.Size([128, 10])
        # print('weighted loss,weighted loss  shape: ',weighted_loss, weighted_loss.shape)
        loss = one_hot_target*weighted_loss
        loss = torch.mean(loss)

        return loss
