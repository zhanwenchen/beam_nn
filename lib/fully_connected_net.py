from torch import nn
import torch
import os
import warnings


class FullyConnectedNet(nn.Module):
    """Fully connected network. ReLU is the activation function.
        Network parameters are intialized with a normal distribution.

    Args:
        input_dim
        output_dim
        layer_width
        num_hidden
        dropout
        dropout_input

    """
    def __init__(self, input_dim, output_dim, layer_width, num_hidden=1, dropout=0, dropout_input=0, batch_norm=False):

        print('fully_connected_net: got input_dim = %s, output_dim = %s, layer_width = %s, num_hidden = %s, dropout = %s, dropout_input = %s, batch_norm = %s' % (input_dim, output_dim, layer_width, num_hidden, dropout, dropout_input, batch_norm))
        super().__init__()

        self.batch_norm = batch_norm

        # input connects to first hidden layer
        self.layers = nn.ModuleList([nn.Linear(input_dim, layer_width)])
        for i in range(num_hidden - 1):
            self.layers.append(nn.Linear(layer_width, layer_width))
        # last hidden connects to output layer
        self.layers.append(nn.Linear(layer_width, output_dim))


        # build as many batch_norm layers minus the last one
        # TODO: assume there's no output batch norm
        if self.batch_norm == True:
            self.batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(layer_width)]) # TODO: not input_dim?
            for i in range(num_hidden - 1):
                self.batch_norm_layers.append(nn.BatchNorm1d(layer_width))
        else:
            warnings.warn('fully_connected_net: not using batch_norm.')


        # activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) # TODO: is dropout actually reusable?
        self.dropout_input = nn.Dropout(dropout_input)


        # initialize weights
        self._initialize_weights()

    def forward(self, x):
        # input dropout
        x = self.dropout_input(x) # REVIEW: Necessary when used after conv1?

        for i in range( len(self.layers) - 1 ):
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
            nn.init.kaiming_normal( self.layers[i].weight.data )
            self.layers[i].bias.data.fill_(0.01) # TODO: Why 0.01 instead of 0?
