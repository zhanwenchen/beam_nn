from torch import nn
import torch
import os

import warnings


class FullyConnectedNet(nn.Module):
    """Fully connected network. ReLU is the activation function.
        Network parameters are intialized with a normal distribution.
    Args:
        input_size
        output_size
        fcs_num_hidden_layers
        num_hidden_layers
        dropout
        dropout_input
    """
    def __init__(self, input_size,
                       output_size,

                       fcs_dropout,
                       batch_norm,

                       fcs_hidden_size,
                       fcs_num_hidden_layers):

        # print('fully_connected_net: got input_size = %s, output_size = %s, fcs_hidden_size = %s, num_hidden_layers = %s, fcs_dropout = %s, fcs_dropout_input = %s, batch_norm = %s' % (input_size, output_size, fcs_hidden_size, num_hidden_layers, fcs_dropout, fcs_dropout_input, batch_norm))
        super().__init__()

        self.batch_norm = batch_norm

        # input connects to first hidden layer
        self.layers = nn.ModuleList([nn.Linear(input_size, fcs_hidden_size)])
        for i in range(fcs_num_hidden_layers - 1):
            self.layers.append(nn.Linear(fcs_hidden_size, fcs_hidden_size))
        # last hidden connects to output layer
        self.layers.append(nn.Linear(fcs_hidden_size, output_size))


        # build as many batch_norm layers minus the last one
        # TODO: assume there's no output batch norm
        if self.batch_norm == True:
            self.batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(fcs_hidden_size)]) # TODO: not input_size?
            for i in range(fcs_num_hidden_layers - 1):
                self.batch_norm_layers.append(nn.BatchNorm1d(fcs_hidden_size))
        else:
            warnings.warn('fully_connected_net: not using batch_norm.')


        # activation and fcs_dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(fcs_dropout) # TODO: is dropout actually reusable?
        # self.dropout_input = nn.Dropout(dropout_input)


        # initialize weights
        self._initialize_weights()

    def forward(self, x):
        # input dropout
        # x = self.dropout_input(x) # REVIEW: Necessary when used after conv1?

        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            if self.batch_norm == True:
                x = self.batch_norm_layers[i](x)
            x = self.relu(x)
            x = self.dropout(x) # is dropout learnable?

        # no dropout or activtion function on the last layer
        x = self.layers[-1](x)
        # TODO: assume there's no output batch norm

        return x

    def _initialize_weights(self):
        for i in range(len(self.layers)):
            nn.init.kaiming_normal_(self.layers[i].weight.data)
            self.layers[i].bias.data.fill_(0.01) # TODO: Why 0.01 instead of 0?
