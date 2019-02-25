import os
import json
import shutil
import errno


from .lenet import LeNet


__all__ = ['get_which_model_from_params_fname', 'read_model_params', 'ensure_dir', 'add_suffix_to_path']


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
    if not os.path.exists(model_params_fname):
        raise OSError('utils.read_model_params: {} doesn\'t exist'.format(model_params_fname))

    with open(model_params_fname, 'r') as f:
        if model_params_fname.endswith('.json'):
            try:
                # model_params = json.load(f, object_hook=_decode)
                model_params = json.load(f)
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


def get_which_model_from_params_fname(model_class, model_params_fname, return_params=False):
    # load the model
    model_params = read_model_params(model_params_fname)
    if 'input_channel' in model_params:
        input_channel = model_params['input_channel']
    else:
        input_channel = 2 # By default, we used 2-channel (2*65) input

    try:
        model = model_class(input_channel,
                            # model_params['input_size'],
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



# https://stackoverflow.com/a/1994840/3853537
def copy_anything(src, dst):
    if os.path.exists(dst):
        print('utils.py: {} exists'.format(dst))
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


def get_pooling_output_dims(input_dims, kernel_dims, stride_dims):
    '''
    Calculate pooling layer output sizes, according to
    http://cs231n.github.io/convolutional-networks/

    W2=(W1−F)/S+1
    H2=(H1−F)/S+1
    D2=D1
    '''
    if isinstance(stride_dims, int, float):
        stride_dims = (stride_dims, stride_dims)

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


def get_conv_output_dims(input_dims, pad_dims, kernel_dims, stride_dims):
    '''
    Calculate pooling layer output sizes, according to
    http://cs231n.github.io/convolutional-networks/

    W2=(W1−F+2P)/S+1
    H2=(H1−F+2P)/S+1 (i.e. width and height are computed equally by symmetry)
    D2=K
    '''
    if isinstance(stride_dims, int, float):
        stride_dims = (stride_dims, stride_dims)

    try:
        conv_input_width, conv_input_height, conv_input_dims = input_dims
        conv_pad_width, conv_pad_height = pad_dims
        conv_kernel_width, conv_kernel_height = kernel_dims
        conv_stride_width, conv_stride_height = stride_dims
    except:
        raise ValueError('{}.get_conv_output_dims: inputs must be ((W_in, H_in, D_in), (W_kernel, H_kernel), (W_stride, H_stride))'.format(__name__))

    conv_output_width = (conv_input_width - conv_kernel_width + 2 * conv_pad_width)/conv_stride_width + 1
    conv_output_height = (conv_input_height - conv_kernel_height + 2 * conv_pad_height)/conv_stride_height + 1
    conv_output_dims = conv_input_dims

    return conv_output_width, conv_output_height, conv_output_dims
