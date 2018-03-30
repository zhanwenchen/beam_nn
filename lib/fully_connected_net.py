from torch import nn

from lib.list_module import ListModule

class FullyConnectedNet(nn.Module):
    def __init__(self, input_dim, output_dim, layer_width, num_hidden_layers):
        """
        Fully connected network with. There are five hidden layers.
        ReLU is the activation function. Network parameters are intialized
        with a normal distribution.

        Args:
            input_dim
            output_dim
            layer_width
        """
        super().__init__()

        self.layers = ListModule(self, 'fc')
        self.layers.append(nn.Linear(input_dim, layer_width))
        for i in range(num_hidden_layers):
            self.layers.append(nn.Linear(layer_width, layer_width))
        self.layers.append(nn.Linear(layer_width, output_dim))

        self.relu = nn.ReLU()
        self._initialize_weights()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        x = self.layers[-1](x)
        return x

    def _initialize_weights(self):
        for layer in self.layers:
            nn.init.kaiming_normal(layer.weight.data)
            layer.bias.data.fill_(0)
