#!/usr/bin/env python
# coding: utf-8

# In[1]:


from json import load
from pprint import pprint

from torch.nn import Conv2d, MaxPool2d, AdaptiveAvgPool2d, ReLU, Sequential, Module

from lib.fully_connected_net import FullyConnectedNet
from lib.flatten import Flatten
from lib.print_layer import PrintLayer


# In[2]:


# fname = 'DNNs/alexnet_2_65_120190602141824593525_created/k_4/model_params.json'


class FlexNet(Module):
    def __init__(self, model_params_fname, printing=False):
        super(FlexNet, self).__init__()
        # Reqad
        with open(model_params_fname) as json_file:
            model_params = load(json_file)

        if printing:
            pprint(model_params)

        layers = model_params['layers']
        modules = []

        for layer in layers:

            if layer['type'] not in ['conv2d', 'maxpool2d', 'adaptiveavgpool2d', 'fcs']:
                raise ValueError('Layer type of {} is not yet implemented'.format(layer['type']))

            if layer['type'] == 'conv2d':
                module = Conv2d(layer['in_channels'],
                                layer['out_channels'],
                                (layer['kernel_height'], layer['kernel_width']),
                                stride=(layer['stride_height'], layer['stride_width']),
                                padding=(layer['padding_height'], layer['padding_width'])
                               )
            if layer['type'] == 'maxpool2d':
                module = MaxPool2d((layer['kernel_height'], layer['kernel_width']),
                                   stride=(layer['stride_height'], layer['stride_width']),
                                  )

            if layer['type'] == 'adaptiveavgpool2d':
                module = AdaptiveAvgPool2d((layer['out_height'], layer['out_width']))

            if layer['type'] == 'fcs':
                flatten = Flatten()
                modules.append(flatten)
                fcs_dropout = 0
                batch_norm = True
                module = FullyConnectedNet(layer['input_size'],
                                           layer['output_size'],
                                           fcs_dropout,
                                           batch_norm,
                                           layer['width'],
                                           layer['num_layers'])

            if printing:
                modules.append(PrintLayer())
            modules.append(module)

            if layer['type'] == 'conv2d':
                modules.append(ReLU(inplace=True))

            net = Sequential(*modules)
            self.net = net
            self.input_dims = model_params['input_dims']

    def forward(self, x):
        input_height, input_width, input_num_channels = self.input_dims
        x = x.view(-1, input_num_channels, input_height, input_width)
        return self.net(x)