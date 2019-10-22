'''
Return dirpath
'''
from os import mkdir as os_mkdir
from os.path import join as os_path_join, isdir as os_path_isdir
from datetime import datetime as datetime_datetime

def create_one_model_keras(models_dirname, model_type, model_version, model_index):
    '''
    Args:
        models_dirname: str, the name of the directory containing
                        all neural network models. It's a short dirname
                        relative to the project root.
        model_type: str, denoting the type of the model. E.g. alexnet, mlp.
        model_version: int,
    Returns:
        dirpath: str. The dirpath of the created model directory.
    '''
    # 1. Come up with a new model name
    timestamp = datetime_datetime.now().strftime('%Y%m%d%H%M%S')
    model_dirname = '{}_v{}_{}_{}_created'.format(model_type, model_version, timestamp, model_index)
    model_dirpath = os_path_join(models_dirname, model_dirname)
    if os_path_isdir(model_dirpath):
        raise OSError('{}: model folder {} should not exist, but it does'.format(__name__, model_dirpath))

    # 2. Make that directory
    os_mkdir(model_dirpath)

    return model_dirpath
