from torch import nn
import torch
import os



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
    def __init__(self, input_dim, output_dim, layer_width, num_hidden=1, dropout=0, dropout_input=0):

        super().__init__()

        # input connects to first hidden layer
        self.layers = nn.ModuleList([nn.Linear(input_dim, layer_width)])

        # other hidden layers
        for i in range(num_hidden-1):
            self.layers.append(nn.Linear(layer_width, layer_width))

        # last hidden connects to output layer
        self.layers.append(nn.Linear(layer_width, output_dim))

        # activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_input = nn.Dropout(dropout_input)

        # initialize weights
        self._initialize_weights()

    def forward(self, x):

        # input dropout
        x = self.dropout_input(x) # REVIEW: Necessary when used after conv1?

        for i in range( len(self.layers) -1 ):
            x = self.dropout( self.relu( self.layers[i](x) ) )

        # no dropout or activtion function on the last layer
        x = self.layers[-1](x)

        return x

    def _initialize_weights(self):

        for i in range(len(self.layers)):
            nn.init.kaiming_normal( self.layers[i].weight.data )
            self.layers[i].bias.data.fill_(0.01)
