import os
import torch

def read_model_params(fname):
    """Read model params from text file
    """
    f = open(fname, 'r')
    model_param_dict = {}
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
        model_param_dict[key] = value  
    f.close()
    return model_param_dict



def save_model_params(fname, model_params_dict):
    """ Save model params to a text file
    """        
    f = open(fname, 'w')
    for key, value in model_params_dict.items():
        print(','.join([str(key), str(value)]), file=f)
    f.close()




def ensure_dir(path):
    """ Check if directory exists. If not, create it
    """
    if not os.path.exists(path):
        os.makedirs(path)




def add_suffix_to_path(path, suffix):
    """ Add suffix to model path
    """
    save_dir = path.split('/')[-2] + suffix
    path = path.split('/')
    path[-2] = save_dir
    path = os.path.join(*path)

    return path

