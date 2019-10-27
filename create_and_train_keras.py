from os import rename as os_rename
import argparse

import plaidml # pylint: disable=E0401

from lib.create_one_model_keras import create_one_model_keras
from lib.train_one_model_keras import train_one_model_keras


MODELS_DIRNAME = 'DNNs'
MODEL_TYPE = 'keras_mlp'
MODEL_VERSION = 0

def create_and_train(how_many, models_dirname=MODELS_DIRNAME, model_type=MODEL_TYPE, model_version=MODEL_VERSION):
    for model_index in range(how_many):
        create_and_train_one(models_dirname, model_type, model_version, model_index)

def create_and_train_one(models_dirname, model_type, model_version, model_index):
    '''
    Args:
        models_dirname is probably DNNS/
    '''
    print('create_and_train_one: models_dirname={}, model_type={}, model_version={}, model_index={}'.format(models_dirname, model_type, model_version, model_index))
    model_dirpath = create_one_model_keras(models_dirname, model_type, model_version, model_index)

    new_model_folder_name = model_dirpath.replace('_created', '_training')
    os_rename(model_dirpath, new_model_folder_name)
    train_one_model_keras(new_model_folder_name)

    os_rename(new_model_folder_name, new_model_folder_name.replace('_training', '_trained'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('how_many', type=int, help='')
    args = parser.parse_args()

    how_many = args.how_many
    create_and_train(how_many)


if __name__ == '__main__':
    # seed_everything()
    main()
