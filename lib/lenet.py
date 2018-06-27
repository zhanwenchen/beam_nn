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


from fully_connected_net import FullyConnectedNet


class LeNet(nn.Module):
    def __init__(self, input_size, output_size, fcs_hidden_size, fcs_num_hidden_layers,
                 pool1_kernel_size,
                 conv1_kernel_size, conv1_num_kernels, conv1_stride,
                 pool2_kernel_size,
                 conv2_kernel_size, conv2_num_kernels, conv2_stride,
                 pool1_stride=2, pool2_stride=2, conv1_dropout=0, conv2_dropout=0.3, fcs_dropout=0.5):
        """"""
        super(LeNet, self).__init__()

        self.input_size = input_size
        self.hidden_size = fcs_hidden_size
        self.output_size = output_size

        input_channel = 2


        # Conv1
        conv1_output_size = (conv1_num_kernels, (input_size - conv1_kernel_size) / conv1_stride + 1)
        if not conv1_output_size[1].is_integer():
            raise ValueError('lenet: conv1_output_size[1] %s is not an integer.' % conv1_output_size[1])
        conv1_output_size = (conv1_num_kernels, int(conv1_output_size[1]))

        self.conv1 = nn.Conv1d(input_channel, conv1_num_kernels, conv1_kernel_size, stride=conv1_stride) # NOTE: THIS IS CORRECT!!!! CONV doesn't depend on num_features!
        nn.init.kaiming_normal(self.conv1.weight.data)
        self.conv1.bias.data.fill_(0)

        # Pool1
        pool1_output_size = (conv1_num_kernels, (conv1_output_size[1] - pool1_kernel_size) / pool1_stride + 1)
        if not pool1_output_size[1].is_integer():
            raise ValueError('lenet: pool1_output_size[1] %s is not an integer.' % pool1_output_size[1])
            pool1_output_size = (conv1_num_kernels, int((conv1_output_size[1] - pool1_kernel_size) / pool1_stride + 1))

        self.pool1 = nn.MaxPool1d(pool1_kernel_size, stride=pool1_stride) # stride=pool1_kernel_size by default


        self.conv1_drop = nn.Dropout2d(p=conv1_dropout)


        # Conv2
        conv2_output_size = (conv2_num_kernels, (pool1_output_size[1] - conv2_kernel_size) / conv2_stride + 1)
        if not conv2_output_size[1].is_integer():
            raise ValueError('lenet: conv2_output_size[1] is not an integer. conv2_output_size =', conv2_output_size)
            conv2_output_size = (conv2_num_kernels, int(conv2_output_size[1]))

        self.conv2 = nn.Conv1d(conv1_num_kernels, conv2_num_kernels, conv2_kernel_size, stride=conv2_stride) # NOTE: THIS IS CORRECT!!!! CONV doesn't depend on num_features!
        nn.init.kaiming_normal(self.conv2.weight.data)
        self.conv2.bias.data.fill_(0)

        self.conv2_drop = nn.Dropout2d(p=conv2_dropout)

        # Pool2
        pool2_output_size = (conv2_num_kernels, (conv2_output_size[1] - pool2_kernel_size) / pool2_stride + 1)
        if not pool2_output_size[1].is_integer():
            raise ValueError('lenet: pool2_output_size[1] is not an integer. pool2_output_size =', pool2_output_size)
            pool2_output_size = (conv2_num_kernels, int((conv2_output_size[1] - pool2_kernel_size) / pool2_stride + 1))

        self.pool2 = nn.MaxPool1d(pool2_kernel_size, stride=pool2_stride) # stride=pool1_kernel_size by default


        # FCs
        fcs_input_size = pool2_output_size[0] * pool2_output_size[1]
        if not fcs_input_size.is_integer():
            raise ValueError('lenet: fcs_input_size = ' + fcs_input_size + ' is not an integer')
        fcs_input_size = int(fcs_input_size)
        self.fcs = FullyConnectedNet(fcs_input_size, output_size, fcs_hidden_size, num_hidden=fcs_num_hidden_layers, dropout=fcs_dropout, batch_norm=True)


    # CHANGED: moved relu after pool
    # CHANGED: moved dropout after relu
    def forward(self, x):
        # pytorch.conv1d accepts shape (Batch, Channel, Width)
        # pytorch.conv2d accepts shape (Batch, Channel, Height, Width)
        # import code; code.interact(local=dict(globals(), **locals()))
        N_elements = int(x.shape[1] / 2)
        x = x.view(-1, 2, N_elements)

        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)
        x = self.conv1_drop(x)
        # x = self.batch_norm1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = F.relu(x)
        x = self.conv2_drop(x)
        # x = self.batch_norm2(x)

        x = x.view(-1, x.size(1) * x.size(2))

        x = self.fcs.forward(x)

        return x
