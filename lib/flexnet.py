from pprint import pprint

from torch.nn import Conv1d, Conv2d, MaxPool2d, AdaptiveAvgPool2d, ReLU, Sequential, Module, Upsample

from lib.fully_connected_net import FullyConnectedNet
from lib.flatten import Flatten
from lib.print_layer import PrintLayer
# from lib.utils import get_dict_from_file_json


class FlexNet(Module):
    # def __init__(self, model_params_fname, printing=False):
    def __init__(self, model_init_params, printing=False):
        super(FlexNet, self).__init__()
        # model_params = get_dict_from_file_json(model_params_fname)
        input_height, _, _, = self.input_dims = model_init_params['input_dims']

        if printing is True:
            pprint(model_init_params)

        layers = model_init_params['layers']
        modules = []

        for layer in layers:

            if layer['type'] == 'upsample':
                if input_height == 2:
                    module = Upsample(scale_factor=(layer['scale_factor_height'], layer['scale_factor_width']))
                elif input_height == 1:
                    module = Upsample(scale_factor=layer['scale_factor_width'])

            if layer['type'] not in ['conv1d', 'upsample', 'conv2d', 'maxpool2d', 'adaptiveavgpool2d', 'fcs']:
                raise ValueError('Layer type of {} is not yet implemented'.format(layer['type']))

            if layer['type'] == 'conv1d':
                module = Conv1d(layer['in_channels'],
                                layer['out_channels'],
                                layer['kernel_width'],
                                stride=layer['stride_width'],
                                padding=layer['padding_width'],
                               )

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

            if printing is True:
                modules.append(PrintLayer(module))
            modules.append(module)

            if layer['type'] == 'conv2d':
                modules.append(ReLU(inplace=True))

            net = Sequential(*modules)
            self.net = net
            self.printing = printing
            # self.model = model_init_params['model']

    def forward(self, x):
        # batch_size = x.size(0)
        input_height, input_width, input_depth = self.input_dims
        if self.printing: print('FlexNet: initial x.size() = ', x.size())
        if input_height == 1:
            # 1D
            x = x.view(-1, input_depth, input_width)
        elif input_height == 2:
            x = x.view(-1, input_depth, input_height, input_width)
        # x = x.view(-1, input_height, input_width, input_num_channels)
        x = self.net(x)
        # if self.model == 'FCN':
        #     x = self.view(batch_size, -1)
        if input_height == 1:
            # 1D
            x = x.view(-1, input_depth * input_width)
        elif input_height == 2:
            x = x.view(-1, input_depth * input_height * input_width)
        return x
