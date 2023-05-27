"""
Vanilla CNN model.
"""

import torch
import torch.nn as nn


class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and padding 0                            #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################
        # CNN : Conv Layer--> ReLU--> MAX pooling --> FC layer
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.conv = nn.Conv2d(in_channels = 3, # input to  network is CIFAR-10 images (32x32 color images)
                               out_channels = 32,
                               kernel_size = (7,7),
                               stride=1,
                               padding=0)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size= (2,2),
                                    stride= 2
                                    )
        # output volume = (h+2P-K)/S+1)*(w+2P-K)/S+1)* input channel size
        # https://discuss.pytorch.org/t/linear-layer-input-neurons-number-calculation-after-conv2d/28659/4
        self.conv_output_size= int((32-7)+1)                                        #26
        self.max_output_size = int((self.conv_output_size-2)/2+1)                   #13
        self.fc_input_size = int(self.max_output_size*self.max_output_size*32)      #13*13*32
        self.fc = nn.Linear(self.fc_input_size, 10)


    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        outs = self.conv(x)   #x shape: torch.Size([128, 3, 32, 32]):[batch_size, channel, height, width].
        # print('x shape:', x.shape)
        # H = (32-7)+1 =26 : [batch size, output channel, H, W]
        # print('conv layer output shape:', outs.shape) #conv layer output shape: torch.Size([128, 32, 26, 26])
        outs = self.relu(outs)
        # print('ReLU output shape:', outs.shape) #ReLU output shape: torch.Size([128, 32, 26, 26])
        outs = self.maxpool(outs)
        # print('MaxPooling output shape:', outs.shape)  # torch.Size([128, 32, 13, 13])
        outs = self.fc(outs.view(-1, self.fc_input_size))

        return outs
