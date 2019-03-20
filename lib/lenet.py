# import warnings
'''
TODO: check stride correctness.

5-layer neural network for
Module calculates convolutional and pooling layer sizes according to
http://cs231n.github.io/convolutional-networks/.


In a , 65 (W) x 4 (H=2, P=2) x 1 (D).
'''

import torch.nn.functional as F
# import torch.nn as nn
from torch import nn, prod
from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, Dropout2d

from lib.fully_connected_net import FullyConnectedNet
from lib.utils import get_pool_output_dims, get_conv_output_dims


printing = False


class LeNet(nn.Module):
    def __init__(self, input_channel,
                       output_size,

                       batch_norm,

                       use_pooling,
                       pooling_method,

                       conv1_kernel_width,
                       conv1_num_kernels,
                       conv1_stride,
                       conv1_dropout,

                       pool1_kernel_size,
                       pool1_stride,

                       conv2_kernel_size,
                       conv2_num_kernels,
                       conv2_stride,
                       conv2_dropout,

                       pool2_kernel_size,
                       pool2_stride,

                       fcs_hidden_size,
                       fcs_num_hidden_layers,
                       fcs_dropout):

        super(LeNet, self).__init__()

        # Instance attributes for use in self.forward() later.
        self.input_channel = input_channel
        self.batch_norm = batch_norm
        try:
            assert isinstance(output_size, int)
        except AssertionError:
            raise AssertionError('{}: output_size should be a Python integer. Got {} of type {}'.format(__name__, output_size, type(output_size)))

        input_size = output_size / self.input_channel
        if input_size.is_integer():
            input_size = int(input_size)
        else:
            raise ValueError('output_size / input_channel = {} / {} = {}'.format(output_size, input_channel, input_size))

        # If not using pooling, set all pooling operations to 1 by 1.
        if use_pooling is False:
            # warnings.warn('lenet: not using pooling')
            pool1_kernel_size = 1
            pool1_stride = 1
            pool2_kernel_size = 1
            pool2_stride = 1

        # #####################################################################
        # Conv1
        # #####################################################################
        conv1_input_width = 65
        conv1_input_height = 2
        conv1_input_depth = input_channel

        conv1_kernel_width = conv1_kernel_width
        conv1_kernel_height = 2

        conv1_stride_width = conv1_stride
        conv1_stride_height = conv1_kernel_height

        conv1_pad_width = 0
        conv1_pad_height = 1

        conv1_output_width, conv1_output_height, conv1_output_depth = get_conv_output_dims(
            (conv1_input_width, conv1_input_height, conv1_input_depth),
            (conv1_pad_width, conv1_pad_height),
            (conv1_kernel_width, conv1_kernel_height),
            (conv1_stride_width, conv1_stride_height),
            conv1_num_kernels)

        # conv1_output_size = (conv1_num_kernels, (input_size - conv1_kernel_width) / conv1_stride + 1)
        conv1_output_size = (conv1_output_width, conv1_output_height, conv1_output_depth)
        if printing: print('lenet.__init__: conv1_output_size (W,H,D) =', conv1_output_size)

        # self.conv1 = nn.Conv1d(input_channel, conv1_num_kernels, conv1_kernel_width, stride=conv1_stride) # NOTE: THIS IS CORRECT!!!! CONV doesn't depend on num_features!
        # pytorch.conv2d accepts shape (Batch (assumed), Channel, Height, Width) - ((Batch, 1, 2, 65))
        # https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d
        self.conv1 = Conv2d(input_channel, conv1_num_kernels,
            (conv1_kernel_height, conv1_kernel_width),
            stride=(conv1_stride_height, conv1_stride_width),
            padding=(conv1_pad_height, conv1_pad_width))
        nn.init.kaiming_normal_(self.conv1.weight.data)
        self.conv1.bias.data.fill_(0)

        self.conv1_drop = Dropout2d(p=conv1_dropout)
        if batch_norm is True:
            self.batch_norm1 = BatchNorm2d(conv1_num_kernels)
        # #####################################################################


        # #####################################################################
        # Pool1
        # #####################################################################
        pool1_input_width = conv1_output_width
        pool1_input_height = conv1_output_height
        pool1_input_depth = conv1_output_depth

        pool1_kernel_width = pool1_kernel_size
        pool1_kernel_height = 1

        pool1_stride_width = pool1_stride
        pool1_stride_height = pool1_kernel_height

        pool1_output_width, pool1_output_height, pool1_output_depth = get_pool_output_dims(
            (pool1_input_width, pool1_input_height, pool1_input_depth),
            (pool1_kernel_width, pool1_kernel_height),
            (pool1_stride_width, pool1_stride_height))

        pool1_output_size = (pool1_output_width, pool1_output_height, pool1_output_depth)
        if printing: print('lenet.__init__: pool1_output_size (W,H,D) =', pool1_output_size)

        # self.pool1 = nn.MaxPool1d(pool1_kernel_size, stride=pool1_stride) # stride=pool1_kernel_size by default
        self.pool1 = MaxPool2d(pool1_kernel_size, stride=pool1_stride) # stride=pool1_kernel_size by default
        #######################################################################


        # #####################################################################
        # Conv2
        # #####################################################################
        conv2_input_width = pool1_output_width
        conv2_input_height = pool1_output_height
        conv2_input_depth = pool1_output_depth

        conv2_kernel_width = conv2_kernel_size
        conv2_kernel_height = 2

        conv2_stride_width = conv2_stride
        conv2_stride_height = conv2_kernel_height

        conv2_pad_width = 0
        conv2_pad_height = 0

        conv2_output_width, conv2_output_height, conv2_output_depth = get_conv_output_dims(
            (conv2_input_width, conv2_input_height, conv2_input_depth),
            (conv2_pad_width, conv2_pad_height),
            (conv2_kernel_width, conv2_kernel_height),
            (conv2_stride_width, conv2_stride_height),
            conv2_num_kernels)
        conv2_output_size = (conv2_output_width, conv2_output_height, conv2_output_depth)

        if printing: print('lenet.__init__: conv2_output_size (W,H,D) =', conv2_output_size)

        # self.conv2 = nn.Conv1d(conv1_num_kernels, conv2_num_kernels, (2, conv2_kernel_size), stride=conv2_stride) # NOTE: THIS IS CORRECT!!!! CONV doesn't depend on num_features!
        self.conv2 = Conv2d(conv1_num_kernels, conv2_num_kernels,
            (conv2_kernel_height, conv2_kernel_width),
            stride=(conv2_stride_height, conv2_stride_width),
            padding=(conv2_pad_height, conv2_pad_width))
        nn.init.kaiming_normal_(self.conv2.weight.data)
        self.conv2.bias.data.fill_(0)

        self.conv2_drop = Dropout2d(p=conv2_dropout)
        if batch_norm is True:
            self.batch_norm2 = BatchNorm2d(conv2_num_kernels)
        # #####################################################################


        # #####################################################################
        # Pool2
        # #####################################################################
        pool2_input_width = conv2_output_width
        pool2_input_height = conv2_output_height
        pool2_input_depth = conv2_output_depth

        pool2_kernel_width = pool2_kernel_size
        pool2_kernel_height = 1

        pool2_stride_width = pool2_stride
        pool2_stride_height = pool2_kernel_height

        pool2_output_width, pool2_output_height, pool2_output_depth = get_pool_output_dims(
            (pool2_input_width, pool2_input_height, pool2_input_depth),
            (pool2_kernel_width, pool2_kernel_height),
            (pool2_stride_width, pool2_stride_height))

        pool2_output_size = (pool2_output_width, pool2_output_height, pool2_output_depth)
        if printing: print('lenet.__init__: pool2_output_size (W,H,D) =', pool2_output_size)

        self.pool2 = MaxPool2d(pool2_kernel_size, stride=pool2_stride) # stride=pool1_kernel_size by default
        #######################################################################

        #
        # # Pool2
        # pool2_output_size = (conv2_num_kernels,
        #         (conv2_output_size[1] - pool2_kernel_size) / pool2_stride + 1)
        #
        # # self.pool2 = nn.MaxPool1d(pool2_kernel_size, stride=pool2_stride) # stride=pool1_kernel_size by default
        # self.pool2 = MaxPool2d(pool2_kernel_size, stride=pool2_stride) # stride=pool1_kernel_size by default

        # FCs
        # fcs_input_size = pool2_output_size[0] * pool2_output_size[1]
        fcs_input_size = pool2_output_width * pool2_output_height * pool2_output_depth
        self.fcs = FullyConnectedNet(fcs_input_size,
                                     output_size,

                                     fcs_dropout,
                                     batch_norm,

                                     fcs_hidden_size,
                                     fcs_num_hidden_layers)


    def forward(self, x):
        # CHANGED: moved relu after pool
        # CHANGED: moved dropout after relu
        # pytorch.conv1d accepts shape (Batch, Channel, Width)
        # pytorch.conv2d accepts shape (Batch, Channel, Height, Width) - ((Batch, 1, 2, 65))
        # import code; code.interact(local=dict(globals(), **locals()))
        # print('lenet: init x.size() =', x.size())
        # num_elements = int(x.shape[1] / self.input_channel)
        # print('lenet: num_elements =', num_elements)
        # x = x.view(-1, 2, num_elements)
        if printing: print('x.size (initial) =', x.size())
        x = x.view(-1, 1, 2, 65)
        # (Batch, 2, 65)
        # print('lenet: after x.view, x.size() =', x.size())

        if printing: print('x.size (after view) =', x.size())

        x = self.conv1(x)
        if printing: print('x.size (after conv1) =', x.size())

        # import code; code.interact(local=dict(globals(), **locals()))

        x = self.pool1(x)
        if printing: print('x.size (after pool1) =', x.size())
        x = F.relu(x)
        if printing: print('x.size (after relu) =', x.size())
        x = self.conv1_drop(x)
        if self.batch_norm is True:
            x = self.batch_norm1(x)

        x = self.conv2(x)
        if printing: print('x.size (after conv2) =', x.size())
        x = self.pool2(x)
        if printing: print('x.size (after pool2) =', x.size())
        x = F.relu(x)
        x = self.conv2_drop(x)
        if self.batch_norm is True:
            x = self.batch_norm2(x)

        # x = x.view(-1, x.size(1) * x.size(2)) # TODO: Multiply another dim
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3)) # TODO: Multiply another dim
        if printing: print('x.size (before fcs) =', x.size())

        x = self.fcs.forward(x)

        return x
