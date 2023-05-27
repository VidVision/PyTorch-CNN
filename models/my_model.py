# Ref:
# https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
# https://discuss.pytorch.org/t/linear-layer-input-neurons-number-calculation-after-conv2d/28659/4
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

"""
MyModel model.
"""


import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        # (Conv Layer--> ReLU--> MAX pooling) -->(Conv Layer--> ReLU--> MAX pooling) --> (Conv Layer--> ReLU--> MAX pooling)
        # (Conv Layer--> ReLU--> MAX pooling) --> (FC layer --> ReLU)--> dropout --> (FC layer --> ReLU) --> (FC layer --> ReLU)
        # --> FC layer
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # https://discuss.pytorch.org/t/linear-layer-input-neurons-number-calculation-after-conv2d/28659/4
        # output volume = (h+2P-K)/S+1)*(w+2P-K)/S+1)* input channel size
        self.conv1 = nn.Conv2d(in_channels=3,  # input to  network is CIFAR-10 images (32x32 color images)
                              out_channels=32,
                              kernel_size=(3, 3),
                              stride=1,
                              padding=1)
        # self.bn1 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2),
                                    stride=2
                                    )
        self.conv2 = nn.Conv2d(in_channels=32,
                              out_channels=64,
                              kernel_size=(3, 3),
                              stride=1,
                              padding=1)
        self.conv3 = nn.Conv2d(in_channels=64,
                              out_channels=128,
                              kernel_size=(3, 3),
                              stride=1,
                              padding=1)
        self.conv4 = nn.Conv2d(in_channels=128,
                              out_channels=256,
                              kernel_size=(3, 3),
                              stride=1,
                              padding=1)
        self.relu = nn.ReLU()
        # self.fc1 = nn.Linear(64*4*4, 512)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.25)
    #     #############################################################################
    #     TODO: Implement forward pass of the network                               #
    #     ############################################################################
    def forward(self, x):
        outs = None
        # add 4 convolution layers followed by activation and pooling
        outs = self.maxpool(self.relu(self.conv1(x)))
        outs = self.maxpool(self.relu(self.conv2(outs)))
        outs = self.maxpool(self.relu(self.conv3(outs)))
        outs = self.maxpool(self.relu(self.conv4(outs)))

        # outs = outs.view(-1, 64 * 4 * 4)
        outs = outs.view(outs.size(0),-1)
        # print('\n \nfc1 input shape:', outs.shape)
        outs = self.relu(self.fc1(outs))
        outs = self.dropout(outs)
        # print('\n \nfc2 input shape:', outs.shape)
        outs = self.relu(self.fc2(outs))
        # print('\n \nfc3 input shape:', outs.shape)
        outs = self.fc3(outs)
        return outs

