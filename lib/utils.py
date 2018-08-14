import os
import torch
from torch.cuda import is_available as get_cuda_available
import json

from .lenet import LeNet


__all__ = ['load_model', 'read_model_params', 'ensure_dir', 'add_suffix_to_path']

def read_model_params(model_params_fname):
    """Read and return model params from json (text) file."""
    with open(model_params_fname, 'r') as f:
        if model_params_fname.endswith('.json'):
            try:
                model_params = json.load(f)
            except:
                raise
        elif model_params_fname.endswith('.txt'):
            model_params = {}
            for line in f:
                [key, value] = line.split(',')
                value = value.rstrip()
                if value.isdigit():
                    value = int(value)
                else:
                    try:
                        value = float(value)
                    except:
                        pass
                model_params[key] = value

    return model_params


def save_model_params(model_params_fname, model_params_dict):
    """Save model params to a json (text) file"""
    with open(model_params_fname, 'w') as f:
        json.dump(model_params_dict, f, indent=4)


def ensure_dir(path):
    """Check if directory exists. If not, create it"""
    if not os.path.exists(path):
        os.makedirs(path)


def add_suffix_to_path(path, suffix):
    """Add suffix to model path."""
    save_dir = path.split('/')[-2] + suffix
    path = path.split('/')
    path[-2] = save_dir
    path = os.path.join(*path)

    return path


def load_model(model_params_fname, using_cuda=True):
    # load the model
    model_params = read_model_params(model_params_fname)
    model = LeNet(model_params['input_size'],
                  model_params['output_size'],

                  model_params['batch_norm'],

                  model_params['use_pooling'],
                  model_params['pooling_method'],

                  model_params['conv1_kernel_size'],
                  model_params['conv1_num_kernels'],
                  model_params['conv1_stride'],
                  model_params['conv1_dropout'],

                  model_params['pool1_kernel_size'],
                  model_params['pool1_stride'],

                  model_params['conv2_kernel_size'],
                  model_params['conv2_num_kernels'],
                  model_params['conv2_stride'],
                  model_params['conv2_dropout'],

                  model_params['pool2_kernel_size'],
                  model_params['pool2_stride'],

                  model_params['fcs_hidden_size'],
                  model_params['fcs_num_hidden_layers'],
                  model_params['fcs_dropout'])

    if using_cuda and get_cuda_available():
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(model_params_fname), 'model.dat')))
        model.cuda()
    else:
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(model_params_fname), 'model.dat'), map_location='cpu'))

    model.eval()

    return model
