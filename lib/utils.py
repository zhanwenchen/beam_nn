import os
from json import load, dump
import shutil
import errno
from math import floor


from torch.nn import Conv2d, MaxPool2d, ReLU

EXCLUDE_MODEL_PARAMS_KEYS = {
    'model',
    'cost_function',
    'cost_function',
    'optimizer',
    'learning_rate',
    'momentum',
    'data_is_target',
    'data_train',
    'data_val',
    'batch_size',
    'data_noise_gaussian',
    'weight_decay',
    'patience',
    'cuda',
    'save_initial',
    'k',
    'save_dir',
}


# __all__ = ['get_which_model_from_params_fname', 'read_model_params', 'ensure_dir', 'add_suffix_to_path', 'get_pool_output_dims', 'get_conv_output_dims']


def _decode(o):
    '''
    Optional object_hook for json.loads(object, object_hook=_decode).
    Code copied and adapted from https://stackoverflow.com/a/48401729.
    '''
    if isinstance(o, str):
        try:
            return int(o)
        except Exception as e:
            print('_decode: trying to convert {} into int but encountered exception {}'.format(o, e))
            return o
    elif isinstance(o, dict):
        return {k: _decode(v) for k, v in o.items()}
    elif isinstance(o, list):
        return [_decode(v) for v in o]
    else:
        return o


def read_model_params(model_params_fname):
    """Read and return model params from json (text) file."""
    # print('read_model_params: model_params_fname = {}'.format(model_params_fname))
    if not os.path.exists(model_params_fname):
        raise OSError('utils.read_model_params: {} doesn\'t exist'.format(model_params_fname))

    with open(model_params_fname, 'r') as f:
        if model_params_fname.endswith('.json'):
            try:
                # model_params = json.load(f, object_hook=_decode)
                model_params = load(f)
            except:
                raise
        elif model_params_fname.endswith('.txt'):
            model_params = {}
            for line in f:
                [key, value] = line.split(',')
                value = value.rstrip()

                # Try to read string values
                if isinstance(value, str):
                    # If value can be turned into a float, then it could also
                    # be an integer.
                    try:
                        value = float(value)
                        if value.is_integer():
                            value = int(value)
                    except ValueError:
                        # If value cannot be turned into a float, then it must
                        # not be a number. In this case, don't do anything and
                        # pass it on as string.
                        pass

                # if isinstance(value, (int, float)):
                #     if value.isdigit():
                #         value = int(value)
                #     else:
                #         value = float(value)
                model_params[key] = value

    return model_params


def save_model_params(model_params_fname, model_params_dict):
    """Save model params to a json (text) file"""
    with open(model_params_fname, 'w') as f:
        dump(model_params_dict, f, indent=4)


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


def get_which_model_from_params_fname(model_params_fname, return_params=False):
    # load the model
    model_params = read_model_params(model_params_fname)

    # If the json is a list, it's a FlexNet
    if 'type' in model_params and model_params['type'] == 'alexnet' and model_params['version'] == '1.0':
        from lib.flexnet import FlexNet
        model_class = FlexNet
        model = FlexNet(model_params_fname)
        if return_params is True:
            return model, model_params
        return model

    if 'model' not in model_params:
        # print('get_which_model_from_params_fname: using LeNet')
        from lib.lenet import LeNet # Circular dependency
        model_class = LeNet
    elif model_params['model'] == 'AlexNet':
        # print('get_which_model_from_params_fname: using AlexNet')
        from lib.alexnet import AlexNet
        model_class = AlexNet

    if 'input_channel' in model_params:
        input_channel = model_params['input_channel']
    else:
        input_channel = 2 # By default, we used 2-channel (2*65) input

    if '2018' in model_params_fname:
        from lib.lenet_1d import LeNet_1D # Circular dependency
        model_class = LeNet_1D
    model_params_init = {k:v for k,v in model_params.items() if k not in EXCLUDE_MODEL_PARAMS_KEYS}

    try:
        # if model_class is FlexNet:
        #     model = model_class(**model_params_init)
        # else:
        print('get_which_model_from_params_fname: input_channel =', input_channel)
        model = model_class(input_channel,
                            # model_params['input'],
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
    except Exception as e:
        raise RuntimeError('{}.get_which_model_from_params_fname: unable to instantiate model class {} with model_params = {}\n. Encountered error: {}'.format(__name__, model_class, model_params, e))

    if return_params is True:
        return model, model_params

    return model


def get_which_model_from_params_fname_old(model_class, model_params_fname, return_params=False):
    # load the model
    model_params = read_model_params(model_params_fname)
    if 'input_channel' in model_params:
        input_channel = model_params['input_channel']
    else:
        input_channel = 2 # By default, we used 2-channel (2*65) input

    model = model_class(input_channel,
                        # model_params['input'],
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

    try:
        # model = model_class(**model_params)
        model = model_class(input_channel,
                            # model_params['input'],
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
    except Exception as e:
        raise RuntimeError('{}.get_which_model_from_params_fname_old: unable to instantiate model class {} with model_params = {}\n. Encountered error: {}'.format(__name__, model_class, model_params, e))

    if return_params is True:
        return model, model_params

    return model


# https://stackoverflow.com/a/46014620/3853537
def copy_create_destination(src, dest):
    try:
        shutil.copy(src, dest)
    except IOError as e:
        # ENOENT(2): file does not exist, raised also on missing dest parent dir
        if e.errno != errno.ENOENT:
            raise
        # try creating parent directories
        os.makedirs(os.path.dirname(dest))
        shutil.copy(src, dest)

# https://stackoverflow.com/a/1994840/3853537
def copy_anything(src, dst):
    # if os.path.exists(dst):
        # print('utils.py: {} exists'.format(dst))
        # shutil.rmtree(dst)
        # shutil.copytree(src, dst)
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else:
            raise


# MATLAB convenience function for cleaning buffer before reading the last info
def clean_buffers(out, err):
    out.seek(0)
    out.truncate(0)

    err.seek(0)
    err.truncate(0)


def get_layers_sizes(input_sizes, layers, return_final_layer_only=False):
    '''
    Args:
        input_sizes: a tuple (H_in, W_in, D_in). E.g. (2, 65, 1)
    '''
    # REVIEW: recursion?
    layers_sizes = []
    for i, layer in enumerate(layers):
        # layer = layers[i]
        if i == 0:
            layer_input_sizes = input_sizes
            # or we could create a custom InputLayer object with similar attributes.
            # Alternatively, we can
            # layer_output_sizes = get_layer_output_size(layer_input_sizes, layer)
        else:
            layer_input_sizes = last_layer_output_sizes
            # layer_layer = layers[i-1]
        layer_size = get_layer_output_size(layer_input_sizes, layer)
        layers_sizes.append(layer_size)

        last_layer_output_sizes = layer_size

    if return_final_layer_only is True:
        return layer_size

    return layers_sizes


def get_layer_output_size(layer_input_sizes, layer):
    '''
    Args:
        layer_input_sizes: tuple of (H_in, W_in, D_in). E.g. (2, 65, 1)
    '''
    # print('get_layers_sizes: layer_input_sizes={}'.format(layer_input_sizes))

    if isinstance(layer, Conv2d):
        return get_conv_output_dims(layer_input_sizes, layer.padding, layer.kernel_size, layer.stride, layer.out_channels)

    if isinstance(layer, MaxPool2d):
        return get_pool_output_dims(layer_input_sizes, layer.kernel_size, layer.stride)

    if isinstance(layer, ReLU):
        return layer_input_sizes

    raise ValueError('get_layer_output_size: layer type if {} is not implemented'.format(type(layer)))


def get_pool_output_dims(input_dims, kernel_dims, stride_dims):
    '''
    Calculate pooling layer output sizes, according to
    http://cs231n.github.io/convolutional-networks/

    Inputs must be ((H_in, W_in, D_in), (H_kernel, W_kernel), (H_stride, W_stride)

    W2=(W1−F)/S+1
    H2=(H1−F)/S+1
    D2=D1
    '''
    try:
        pool_input_height, pool_input_width, pool_input_dims = input_dims
        pool_kernel_height, pool_kernel_width = kernel_dims
        pool_stride_height, pool_stride_width = stride_dims
    except:
        raise ValueError('{}.get_pooling_output_dims: inputs must be ((H_in, W_in, D_in), (H_kernel, W_kernel), (H_stride, W_stride)). Got input_dims={}, kernel_dims={}, stride_dims={}'.format(__name__, input_dims, kernel_dims, stride_dims))

    pool_output_height = floor((pool_input_height - pool_kernel_height)/pool_stride_height + 1)
    pool_output_width = floor((pool_input_width - pool_kernel_width)/pool_stride_width + 1)
    # print('{pool_output_width} = floor(({pool_input_width} - {pool_kernel_width})/{pool_stride_width} + 1)'.format(pool_output_width=pool_output_width, pool_input_width=pool_input_width, pool_kernel_width=pool_kernel_width, pool_stride_width=pool_stride_width))
    pool_output_dims = pool_input_dims

    return pool_output_height, pool_output_width, pool_output_dims


def get_pool_output_dims_old(input_dims, kernel_dims, stride_dims):
    '''
    Calculate pooling layer output sizes, according to
    http://cs231n.github.io/convolutional-networks/

    Inputs must be ((W_in, H_in, D_in), (W_kernel, H_kernel), (W_stride, H_stride)

    W2=(W1−F)/S+1
    H2=(H1−F)/S+1
    D2=D1
    '''
    try:
        pool_input_width, pool_input_height, pool_input_dims = input_dims
        pool_kernel_width, pool_kernel_height = kernel_dims
        pool_stride_width, pool_stride_height = stride_dims
    except:
        raise ValueError('{}.get_pooling_output_dims: inputs must be ((W_in, H_in, D_in), (W_kernel, H_kernel), (W_stride, H_stride))'.format(__name__))

    pool_output_width = (pool_input_width - pool_kernel_width)/pool_stride_width + 1
    pool_output_height = (pool_input_height - pool_kernel_height)/pool_stride_height + 1
    pool_output_dims = pool_input_dims

    return pool_output_width, pool_output_height, pool_output_dims


def get_conv_output_dims(input_dims, pad_dims, kernel_dims, stride_dims, num_kernels):
    '''
    Calculate pooling layer output sizes, according to
    http://cs231n.github.io/convolutional-networks/

    Inputs must be ((H_in, W_in, D_in), (H_kernel, W_kernel), (H_stride, W_stride))

    W2=(W1−F+2P)/S+1
    H2=(H1−F+2P)/S+1 (i.e. width and height are computed equally by symmetry)
    D2=K
    '''
    try:
        conv_input_height, conv_input_width, conv_input_depth = input_dims
        conv_pad_height, conv_pad_width = pad_dims
        conv_kernel_height, conv_kernel_width = kernel_dims
        conv_stride_height, conv_stride_width = stride_dims
    except:
        raise ValueError('{}.get_conv_output_dims: inputs must be ((H_in, W_in, D_in), (H_kernel, W_kernel), (H_stride, W_stride))'.format(__name__))

    conv_output_height = floor((conv_input_height - conv_kernel_height + 2 * conv_pad_height)/conv_stride_height + 1)
    conv_output_width = floor((conv_input_width - conv_kernel_width + 2 * conv_pad_width)/conv_stride_width + 1)
    # print('{conv_output_width} = floor(({conv_input_width} - {conv_kernel_width} + 2 * {conv_pad_width})/{conv_stride_width} + 1)'.format(conv_output_width=conv_output_width, conv_input_width=conv_input_width, conv_kernel_width=conv_kernel_width, conv_pad_width=conv_pad_width, conv_stride_width=conv_stride_width))
    # print('conv_output_height = (W − F + 2P)/S + 1 = ({} - {} + 2 x {})/{} + 1 = {}'.format(conv_input_height, conv_kernel_height, conv_pad_height, conv_stride_height, conv_output_height))
    conv_output_dims = num_kernels

    return conv_output_height, conv_output_width, conv_output_dims
