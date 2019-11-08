from pprint import pprint

from torch.nn import Conv1d, Conv2d, MaxPool2d, AdaptiveAvgPool2d, LeakyReLU, Sequential, Module, Upsample, BatchNorm1d, BatchNorm2d, Linear, Dropout

from lib.fully_connected_net import FullyConnectedNet
from lib.flatten import Flatten
from lib.print_layer import PrintLayer


# NOTE: FlexNet is only for regression.
class FlexNet(Module):
    def __init__(self, model_init_params, printing=False):
        super(FlexNet, self).__init__()
        if 'input_dims' in model_init_params:
            input_height, _, _, = self.input_dims = model_init_params['input_dims']
        elif 'input_size' in model_init_params:
            input_size = model_init_params['input_size']

        if printing is True:
            pprint(model_init_params)

        try:
            layers = model_init_params['layers']
        except:
            raise ValueError('Problem reading layers. model_init_params={}'.format(model_init_params))
        num_layers = len(layers)
        modules = []

        # batch_norm_enable = model_init_params['batch_norm_enable']
        if 'dropout' in model_init_params:
            dropout = model_init_params['dropout']
        else:
            dropout = None

        for index, layer in enumerate(layers):
            layer_type = layer['type']
            if layer_type == 'upsample':
                if input_height == 2:
                    module = Upsample(scale_factor=(layer['scale_factor_height'], layer['scale_factor_width']))
                elif input_height == 1:
                    module = Upsample(scale_factor=layer['scale_factor_width'])

            if layer_type not in ['fc', 'conv1d', 'upsample', 'conv2d', 'maxpool2d', 'adaptiveavgpool2d', 'fcs']:
                raise ValueError('Layer type of {} is not yet implemented'.format(layer['type']))

            if layer_type == 'conv1d':
                module = Conv1d(layer['in_channels'],
                                layer['out_channels'],
                                layer['kernel_width'],
                                stride=layer['stride_width'],
                                padding=layer['padding_width'],
                               )

            if layer_type == 'conv2d':
                module = Conv2d(layer['in_channels'],
                                layer['out_channels'],
                                (layer['kernel_height'], layer['kernel_width']),
                                stride=(layer['stride_height'], layer['stride_width']),
                                padding=(layer['padding_height'], layer['padding_width']))

            if layer_type == 'maxpool2d':
                module = MaxPool2d((layer['kernel_height'], layer['kernel_width']),
                                   stride=(layer['stride_height'], layer['stride_width']),
                                  )

            if layer_type == 'adaptiveavgpool2d':
                module = AdaptiveAvgPool2d((layer['out_height'], layer['out_width']))

            if layer_type == 'fc':
                module = Linear(layer['in_channels'], layer['out_channels'])

            if layer_type == 'fcs':
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

            if printing is True:
                modules.append(PrintLayer(module))
            modules.append(module)

            if index != num_layers - 1 and layer_type in ['fc', 'conv1d', 'conv2d']:
                # For layers other than the last layer, add BatchNorm followed by LeakyReLU
                # Last layer of regression should not have nonlinear activation
                layer_out_channels = layer['out_channels']
                if layer_type == 'conv1d':
                    modules.append(BatchNorm1d(layer_out_channels))
                elif layer_type == 'conv2d':
                    modules.append(BatchNorm2d(layer_out_channels))

                modules.append(LeakyReLU())

            # After activation, apply dropout
            if dropout is not None:
                modules.append(Dropout(p=dropout))

            del module

        self.net = Sequential(*modules)
        del modules
        self.printing = printing
        # self.model = model_init_params['model']

    def forward(self, x):
        batch_size = x.size(0)
        # First layer is conv
        if hasattr(self, 'input_dims'):
            input_height, input_width, input_depth = self.input_dims
            if input_height == 1:
                # 1D
                x = x.view(batch_size, input_depth, input_width)
            elif input_height == 2:
                x = x.view(batch_size, input_depth, input_height, input_width)
        # First layer is FC

        if self.printing:
            print('FlexNet: initial x.size() = ', x.size())

        if self.printing:
            print('FlexNet: after first view, x.size() = ', x.size())
        x = self.net(x)
        if hasattr(self, 'input_dims'):
            # BUG: Is this a problem with destoying convolution? This is just flatten for training and test.
            #      Maye this shouldn't be "viewed" at all and the compute cost in trainer should be modified.
            if input_height == 1:
            # 1D
                x = x.view(-1, input_depth * input_width)
            elif input_height == 2:
                x = x.view(-1, input_depth * input_height * input_width)
        if self.printing:
            print('FlexNet: finally x.size() = ', x.size())
        return x
