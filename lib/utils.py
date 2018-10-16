import os
import json
import shutil
import errno


from .lenet import LeNet


__all__ = ['get_which_model_from_params_fname', 'read_model_params', 'ensure_dir', 'add_suffix_to_path']

def read_model_params(model_params_fname):
    """Read and return model params from json (text) file."""
    if not os.path.exists(model_params_fname):
        raise OSError('utils.read_model_params: {} doesn\'t exist'.format(model_params_fname))

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
                        raise
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
    model = model_class(model_params['input_channel'],
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
