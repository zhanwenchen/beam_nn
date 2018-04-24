# This is option 1 of CNN - use one channel (130 * 1)
# TODO: Option 2: use two channels - real vs imaginery components each (65 * 2)

import torch
import torch.nn.functional as F
import torch.nn as nn
from lib.fully_connected_net import FullyConnectedNet

# CNN Model (2 conv layer)
class LeNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_hidden_layers, batch_size, kernel_size=2, num_kernels=1, stride=1):
        """
        kernel_size=2: The size of the sliding window.
        num_kernels=1: Essentially the number of neurons for a conv layer.

        """
        super(LeNet, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        input_channel = 2
        self.conv1 = nn.Conv1d(input_channel, num_kernels, kernel_size) # NOTE: THIS IS CORRECT!!!! CONV doesn't depend on num_features!
        # self.conv1 = nn.Conv2d(input_channel, num_kernels, kernel_size)
        # self.pool1 = nn.AvgPool1d(2)
        # self.conv2 = nn.Conv2d(input_channel, num_kernels, kernel_size)

        # NOTE: Imaginery data can be treated in two ways:
        #      1. as a separate channel, still using conv1d.
        #      2. as a separate dimension, using conv2d.
        # FIXME: How do we have an NN output 2D, or 1D with another channel? 130 by 1 like it used to be.
        # TODO 1. Equal conv, fc layers
        # TODO 2. num_kernels try 16+
        # TODO 3. try bigger strides
        # TODO 4. try adam

        # self.conv2 = nn.Conv1d(1, 1, 2) # CHANGED: let's try one layer for now
        # self.pool2 = nn.AvgPool1d(2)

        # TODO get num_features for any stride
        if stride == 1:
            conv_output_features = input_size - kernel_size + 1

        nn.init.kaiming_normal(self.conv1.weight.data)
        self.conv1.bias.data.fill_(0)
        self.fcs = FullyConnectedNet(conv_output_features, output_size, hidden_size, num_hidden_layers)


    def forward(self, x):
        # pytorch.conv1d accepts shape (Batch, Channel, Width)
        # pytorch.conv2d accepts shape (Batch, Channel, Height, Width)
        # print("x.size() was", x.size())
        # x = x.transpose(1,2)
        # print("x.size() is", x.size())

        # x = x.unsqueeze(2) # right now it's (32, 130, 1). Should be (130, 32, 1) or (1, 32, 130). NOTE it's not (1, 32, 130) or (32, 1, 130).  It has to be (1, 130, 32)
        # print("x.size() unsqueezed", x.size())
        # print("lenet.forward: x.shape =", x.shape)
        x = self.conv1(x)
        # print("lenet.forward: x.shape =", x.shape)
        x = F.relu(x)
        # print("lenet.forward: x.shape =", x.shape)
        x = self.fcs.forward(x)
        # x = self.pool1(x)

        # x = self.conv2(x)
        # x = F.relu(x)
        # x = self.pool2(x)

        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        # print("lenet.forward: x.shape =", x.shape)
        return x
