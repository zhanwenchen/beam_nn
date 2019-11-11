from os.path import join as os_path_join
from copy import copy as copy_copy
import datetime
import argparse

from lib.utils import save_model_params, ensure_dir


MODEL_PARAMS_FNAME = 'model_params.json'
MODEL_DICT_TEMPLATE = {
  "batch_size":32,
  "data_is_target":0,
  "data_noise_gaussian":1,
  "data_train":"data/20180402_L74_70mm/train_2.h5",
  "data_val":"data/20180402_L74_70mm/val_2.h5",
  "input_dims": [1, 130, 1 ],
  "learning_rate":1.0e-5,
  "loss_function":"SmoothL1",
  "model":"FCNN",
  "momentum":0,
  "optimizer":"Adam",
  "patience":20,
  "version":"1.6.7",
  "weight_decay":0
}

LAYER_DICT_TEMPLATE = {
    "kernel_height":1,
    "padding_height":0,
    "stride_height":0,
    "type":"conv1d"
}


def get_layers(kernel_width, num_kernels, num_layers):
    layers = []
    for layer_index in range(num_layers):
        layer = copy_copy(LAYER_DICT_TEMPLATE)

        if layer_index == num_layers - 1:
            # for the last layer, the output needs to be
            layer["out_channels"] = 1
        else:
            layer["out_channels"] = num_kernels

        if kernel_width == 5:
            layer['kernel_width'] = 5
            layer['padding_width'] = 2
            layer['stride_width'] = 1
        elif kernel_width == 7:
            layer['kernel_width'] = 7
            layer['padding_width'] = 3
            layer['stride_width'] = 1

        if layer_index == 0:
            layer['in_channels'] = 1
        else:
            layer['in_channels'] = layers[-1]['out_channels']

        layer['name'] = 'conv{}'.format(layer_index+1)

        layers.append(layer)

    return layers


def get_model_dict(kernel_width, num_kernels, num_layers):
    model_dict = copy_copy(MODEL_DICT_TEMPLATE)
    layers = get_layers(kernel_width, num_kernels, num_layers)
    model_dict['layers'] = layers
    return model_dict

def get_and_save_model_dict(kernel_width, num_kernels, num_layers, index):
    model_dict = get_model_dict(kernel_width, num_kernels, num_layers)
    for k in [3, 4, 5]:
        model_dict['k'] = k
        identifier = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        model_dict['save_dir'] = os_path_join('DNNs', 'fcnn_v1.6.7_{}_{}_created'.format(identifier, index), 'k_{}'.format(k))

        # print(model_params['save_dir'])
        ensure_dir(model_dict['save_dir'])
        save_model_params(os_path_join(model_dict['save_dir'], MODEL_PARAMS_FNAME), model_dict)
        print('created ', model_dict['save_dir'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kernel_width', type=int)
    parser.add_argument('num_kernels', type=int)
    parser.add_argument('num_layers', type=int)
    parser.add_argument('num_models', type=int)
    args = parser.parse_args()

    kernel_width = args.kernel_width
    num_kernels = args.num_kernels
    num_layers = args.num_layers
    num_models = args.num_models
    [get_and_save_model_dict(kernel_width, num_kernels, num_layers, index) for index in range(num_models)]
