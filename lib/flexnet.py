#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from pprint import pprint

from torch.nn import Conv2d, MaxPool2d, AdaptiveAvgPool2d, ReLU, Sequential, Module

from lib.fully_connected_net import FullyConnectedNet
from lib.flatten import Flatten


# In[2]:


fname = 'DNNs/alexnet_2_65_120190602141824593525_created/k_4/model_params.json'


class FlexNet(Module):
    def __init__(self, json_file):


with open(fname) as json_file:
    layers = json.load(json_file)

pprint(layers)


modules = []

for layer in layers:
    if layer['type'] == 'conv2d':
        module = Conv2d(layer['in_channels'],
                        layer['out_channels'],
                        (layer['kernel_height'], layer['kernel_width']),
                        stride=(layer['stride_height'], layer['stride_width'])
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
        module = FullyConnectedNet(layer['input_size'], layer['output_size'], fcs_dropout,                                    batch_norm, layer['width'], layer['num_layers'])


    modules.append(module)

    if layer['type'] == 'conv2d':
        modules.append(ReLU(inplace=True))





    self.net = Sequential(*modules)
