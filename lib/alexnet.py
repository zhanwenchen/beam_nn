# import warnings
'''
Original AlexNet input: 227 x 227.
Reference: https://www.learnopencv.com/number-of-parameters-and-tensor-sizes-in-convolutional-neural-network/



In a , 65 (W) x 4 (H=2, P=2) x 1 (D).
'''
from torch.nn import Module, Conv2d, MaxPool2d, Dropout, Linear, ReLU, AdaptiveAvgPool2d, Sequential

# from lib.fully_connected_net import FullyConnectedNet
from lib.utils import get_layers_sizes


printing = False


# import torch.nn as nn
# import torch.utils.model_zoo as model_zoo


# __all__ = ['AlexNet', 'alexnet']


# model_urls = {
#     'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
# }

# input_num_channels = 1
# input_height = 2
# input_width = 65
#
# conv1_num_kernels = 32
# conv1_kernel_height = 2
# conv1_kernel_width = 9
# conv1_stride_height = 2
# conv1_stride_width = 3
# conv1_padding_height = 1
# conv1_padding_width = 2
#
# pool1_kernel_height = 1
# pool1_kernel_width = 3
# pool1_stride_height = 1
# pool1_stride_width = 2
#
# conv2_num_kernels = 96
# conv2_kernel_height = 2
# conv2_kernel_width = 5
# conv2_stride_height = 2
# conv2_stride_width = 1
# conv2_padding_height = 1
# conv2_padding_width = 2
#
# pool2_kernel_height = 1
# pool2_kernel_width = 3
# pool2_stride_height = 1
# pool2_stride_width = 2
#
# conv3_num_kernels = 192
# conv3_kernel_height = 2
# conv3_kernel_width = 3
# conv3_stride_height = 2
# conv3_stride_width = 1
# conv3_padding_height = 1
# conv3_padding_width = 1
#
# conv4_num_kernels = 64
# conv4_kernel_height = 2
# conv4_kernel_width = 3
# conv4_stride_height = 2
# conv4_stride_width = 1
# conv4_padding_height = 1
# conv4_padding_width = 1
#
# conv5_num_kernels = 64
# conv5_kernel_height = 2
# conv5_kernel_width = 3
# conv5_stride_height = 2
# conv5_stride_width = 1
# conv5_padding_height = 1
# conv5_padding_width = 1
#
# pool3_kernel_height = 1
# pool3_kernel_width = 2
# pool3_stride_height = 1
# pool3_stride_width = 2
#
# avgpool_out_height = 2
# avgpool_out_width = 4
#
# fcs_hidden_size = 4096

# output_size = 130


class AlexNet(Module):
    def __init__(self, output_size=None, input_width=None, input_height=None, input_num_channels=None,
                 conv1_num_kernels=None, conv1_kernel_height=None, conv1_kernel_width=None, conv1_stride_height=None, conv1_stride_width=None, conv1_padding_height=None, conv1_padding_width=None,
                 pool1_kernel_height=None, pool1_kernel_width=None, pool1_stride_height=None, pool1_stride_width=None,
                 conv2_num_kernels=None, conv2_kernel_height=None, conv2_kernel_width=None, conv2_stride_height=None, conv2_stride_width=None, conv2_padding_height=None, conv2_padding_width=None,
                 pool2_kernel_height=None, pool2_kernel_width=None, pool2_stride_height=None, pool2_stride_width=None,
                 conv3_num_kernels=None, conv3_kernel_height=None, conv3_kernel_width=None, conv3_stride_height=None, conv3_stride_width=None, conv3_padding_height=None, conv3_padding_width=None,
                 conv4_num_kernels=None, conv4_kernel_height=None, conv4_kernel_width=None, conv4_stride_height=None, conv4_stride_width=None, conv4_padding_height=None, conv4_padding_width=None,
                 conv5_num_kernels=None, conv5_kernel_height=None, conv5_kernel_width=None, conv5_stride_height=None, conv5_stride_width=None, conv5_padding_height=None, conv5_padding_width=None,
                 pool3_kernel_height=None, pool3_kernel_width=None, pool3_stride_height=None, pool3_stride_width=None,
                 avgpool_out_height=None, avgpool_out_width=None,
                 fcs_hidden_size=None,
                 ):
        super(AlexNet, self).__init__()

        self.input_num_channels, self.input_height, self.input_width = input_num_channels, input_height, input_width

        self.features = Sequential(
            Conv2d(input_num_channels, conv1_num_kernels, kernel_size=(conv1_kernel_height, conv1_kernel_width), stride=(conv1_stride_height, conv1_stride_width), padding=(conv1_padding_height, conv1_padding_width)),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=(pool1_kernel_height, pool1_kernel_width), stride=(pool1_stride_height, pool1_stride_width)),
            Conv2d(conv1_num_kernels, conv2_num_kernels, kernel_size=(conv2_kernel_height, conv2_kernel_width), stride=(conv2_stride_height, conv2_stride_width), padding=(conv2_padding_height, conv2_padding_width)),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=(pool2_kernel_height, pool2_kernel_width), stride=(pool2_stride_height, pool2_stride_width)),
            Conv2d(conv2_num_kernels, conv3_num_kernels, kernel_size=(conv3_kernel_height, conv3_kernel_width), stride=(conv3_stride_height, conv3_stride_width), padding=(conv3_padding_height, conv3_padding_width)),
            ReLU(inplace=True),
            Conv2d(conv3_num_kernels, conv4_num_kernels, kernel_size=(conv4_kernel_height, conv4_kernel_width), stride=(conv4_stride_height, conv4_stride_width), padding=(conv4_padding_height, conv4_padding_width)),
            ReLU(inplace=True),
            Conv2d(conv4_num_kernels, conv5_num_kernels, kernel_size=(conv5_kernel_height, conv5_kernel_width), stride=(conv5_stride_height, conv5_stride_width), padding=(conv5_padding_height, conv5_padding_width)),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=(pool3_kernel_height, pool3_kernel_width), stride=(pool3_stride_height, pool3_stride_width)),
        )
        self.features_output_sizes = get_layers_sizes((input_height, input_width, input_num_channels), self.features, return_final_layer_only=True)
        print('self.features_output_sizes =', self.features_output_sizes)
        self.features_output_prod = self.features_output_sizes[0] * self.features_output_sizes[1] * self.features_output_sizes[2]
        print('self.features_output_prod =', self.features_output_prod)
        # self.avgpool = AdaptiveAvgPool2d((self.features_output_prod[0], self.features_output_prod[1])) # constraining by architecture.
        self.avgpool = AdaptiveAvgPool2d((avgpool_out_height, avgpool_out_width)) # parameterized
        print('fcs_hidden_size = {}, output_size={}'.format(fcs_hidden_size, output_size))
        self.classifier = Sequential(
            Dropout(),
            Linear(self.features_output_prod, fcs_hidden_size),
            ReLU(inplace=True),
            Dropout(),
            Linear(fcs_hidden_size, fcs_hidden_size),
            ReLU(inplace=True),
            Linear(fcs_hidden_size, output_size),
        )

    def forward(self, x):
        if printing: print('x.size (initial) =', x.size())
        x = x.view(-1, self.input_num_channels, self.input_height, self.input_width)
        if printing: print('x.size (initial) =', x.size())
        x = self.features(x)
        if printing: print('x.size (features) =', x.size())
        x = self.avgpool(x)
        if printing: print('x.size (after avgpool) =', x.size())
        x = x.view(x.size(0), self.features_output_prod)
        if printing: print('x.size (after view) =', x.size())
        x = self.classifier(x)
        if printing: print('x.size (after classifier) =', x.size())
        return x
