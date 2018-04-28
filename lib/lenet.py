# NOTE: Imaginery data can be treated in two ways:
#      1. as a separate channel, still using conv1d.
#      2. as a separate dimension, using conv2d.
# FIXME: How do we have an NN output 2D, or 1D with another channel? 130 by 1 like it used to be.
# TODO 1. Equal conv, fc layers
# TODO 2. num_kernels try 16+
# TODO 3. try bigger strides
# TODO 4. try adam

# This is option 1 of CNN - use one channel (130 * 1)
# TODO: Option 2: use two channels - real vs imaginery components each (65 * 2)

import torch
import torch.nn.functional as F
import torch.nn as nn
from lib.fully_connected_net import FullyConnectedNet

# CNN Model (2 conv layer)
class LeNet(nn.Module):
    def __init__(self, input_size, output_size, fcs_hidden_size, fcs_num_hidden_layers,
                 batch_size,
                 pool1_kernel_size,
                 conv1_kernel_size, conv1_num_kernels, conv1_stride,
                 pool2_kernel_size,
                 conv2_kernel_size, conv2_num_kernels, conv2_stride):
        """
        kernel_size=2: The size of the sliding window.
        num_kernels=1: Essentially the number of neurons for a conv layer.

        """
        super(LeNet, self).__init__()

        self.input_size = input_size
        self.hidden_size = fcs_hidden_size
        self.output_size = output_size

        input_channel = 2
        # print("lenet: conv2: conv2_kernel_size = %s, conv2_num_kernels = %s, conv2_stride = %s" % (conv2_kernel_size, conv2_num_kernels, conv2_stride))
        # print("input_size =", input_size)


        # Conv1
        # print("lenet: conv1_kernel_size =", conv1_kernel_size)
        # print("lenet: conv1_stride =", conv1_stride)


        conv1_output_size = (conv1_num_kernels, (input_size - conv1_kernel_size) / conv1_stride + 1)
        # print("lenet: conv1_output_size =", conv1_output_size)
        if not conv1_output_size[1].is_integer(): raise ValueError('lenet: conv1_output_size is not an integer. It is', conv1_output_size)
        conv1_output_size = (conv1_num_kernels, int(conv1_output_size[1]))

        self.conv1 = nn.Conv1d(input_channel, conv1_num_kernels, conv1_kernel_size, stride=conv1_stride) # NOTE: THIS IS CORRECT!!!! CONV doesn't depend on num_features!
        nn.init.kaiming_normal(self.conv1.weight.data)
        self.conv1.bias.data.fill_(0)






        # Pool1
        pool1_stride = 2
        # print("lenet: pool1_kernel_size =", pool1_kernel_size)

        # print("lenet: pool1_output_size = (%s - %s ) / 2 + 1 = %s" % (conv1_output_size[1], pool1_kernel_size, (conv1_output_size[1] - pool1_kernel_size) / pool1_stride + 1))
        pool1_output_size = (conv1_num_kernels, (conv1_output_size[1] - pool1_kernel_size) / pool1_stride + 1)
        if not pool1_output_size[1].is_integer(): raise ValueError('lenet: pool1_kernel_size is not an integer. It is', pool1_output_size)
        pool1_output_size = (conv1_num_kernels, int((conv1_output_size[1] - pool1_kernel_size) / pool1_stride + 1))
        # print("lenet: pool1_output_size =", pool1_output_size)



        self.pool1 = nn.MaxPool1d(pool1_kernel_size, stride=pool1_stride) # stride=pool1_kernel_size by default



        # Conv2


        conv2_output_size = (conv2_num_kernels, (pool1_output_size[1] - conv2_kernel_size) / conv2_stride + 1)
        # print("lenet: conv2_output_size = (%s - %s ) / %s + 1 = %s" % (pool1_output_size[1], conv2_kernel_size, conv2_stride, conv2_output_size[1]))
        # print("lenet: conv2_output_size =", conv2_output_size)
        if not conv2_output_size[1].is_integer(): raise ValueError('lenet: conv2_output_size[1] is not an integer. conv2_output_size =', conv2_output_size)
        conv2_output_size = (conv2_num_kernels, int(conv2_output_size[1]))


        self.conv2 = nn.Conv1d(conv1_num_kernels, conv2_num_kernels, conv2_kernel_size, stride=conv2_stride) # NOTE: THIS IS CORRECT!!!! CONV doesn't depend on num_features!
        nn.init.kaiming_normal(self.conv2.weight.data)
        self.conv2.bias.data.fill_(0)


        # Pool2
        pool2_stride = 2
        pool2_output_size = (conv2_num_kernels, (conv2_output_size[1] - pool2_kernel_size) / pool2_stride + 1)
        # print("lenet: pool2_output_size =", pool2_output_size)
        if not pool2_output_size[1].is_integer(): raise ValueError('lenet: pool2_output_size[1] is not an integer. pool2_output_size =', pool2_output_size)
        pool2_output_size = (conv2_num_kernels, int((conv2_output_size[1] - pool2_kernel_size) / pool2_stride + 1))


        self.pool2 = nn.MaxPool1d(pool2_kernel_size, stride=pool2_stride) # stride=pool1_kernel_size by default


        # FCs
        # conv2_output_size = (conv1_output_size - conv1_kernel_size) / conv2_stride
        # if not pool1_kernel_size.is_integer(): raise ValueError('lenet: pool1_kernel_size is not an integer. It is', pool1_kernel_size)
        fcs_input_size = pool2_output_size[0] * pool2_output_size[1]
        # print("lenet: fcs_input_size =", fcs_input_size)
        self.fcs = FullyConnectedNet(fcs_input_size, output_size, fcs_hidden_size, fcs_num_hidden_layers)


    def forward(self, x):
        # pytorch.conv1d accepts shape (Batch, Channel, Width)
        # pytorch.conv2d accepts shape (Batch, Channel, Height, Width)
        # print("input x.size() =", x.size())
        x = self.conv1(x)
        # print("conv1 x.size() =", x.size())
        x = F.relu(x)
        x = self.pool1(x)
        # print("pool1 x.size() =", x.size())

        x = self.conv2(x)
        # print("conv2 x.size() =", x.size())
        x = F.relu(x)
        x = self.pool2(x)
        # print("pool2 x.size() =", x.size())

        x = x.view(-1, x.size(1) * x.size(2))
        # print("view x.size() =", x.size())

        x = self.fcs.forward(x)
        # print("fcs x.size() =", x.size())
        # print("lenet.forward: x.shape =", x.shape)
        return x
