"""
Two Layer Network Model.
"""

import torch
import torch.nn as nn


class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        """
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        """
        super(TwoLayerNet, self).__init__()
        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################
        # Fully Connected Net : one linear layer --> sigmoid activation --> one linear layer
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.sig = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        out = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################

        out = self.fc1(x.view(x.shape[0],-1))  #reshape the input data # https://pytorch.org/docs/stable/generated/torch.Tensor.view.html#torch.Tensor.view
        out = self.sig(out)
        out = self.fc2(out)

        return out
