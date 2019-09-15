import plaidml

from glob import glob as glob_glob
from os.path import join as os_path_join
from os import rename as os_rename
from functools import partial as functools_partial
from itertools import repeat as itertools_repeat
# from shutil import move as shutil_move
import argparse
from multiprocessing.pool import ThreadPool

from lib.create_one_model_keras import create_one_model_keras
from lib.train_one_model_keras import train_one_model_keras


MODELS_DIRNAME = 'DNNs'
MODEL_TYPE = 'keras_mlp'
MODEL_VERSION = 0

def create_and_train(how_many, models_dirname=MODELS_DIRNAME, model_type=MODEL_TYPE, model_version=MODEL_VERSION):
    for model_index in range(how_many):
        create_and_train_one(models_dirname, model_type, model_version, model_index)
    # with ThreadPool() as pool:
    #     list(pool.imap_unordered(functools_partial(create_and_train_one, models_dirname, model_type, model_version), list(range(how_many))))
        # pool.map(create_and_train_one, list(range(how_many)), itertools_repeat(models_dirname), itertools_repeat(model_type), itertools_repeat(model_version))


def create_and_train_one(models_dirname, model_type, model_version, model_index):
    '''
    Args:
        models_dirname is probably DNNS/
    '''
    print('create_and_train_one: models_dirname={}, model_type={}, model_version={}, model_index={}'.format(models_dirname, model_type, model_version, model_index))
    model_dirpath = create_one_model_keras(models_dirname, model_type, model_version, model_index)
    # models = glob_glob(os_path_join(MODELS_DIRNAME, str(identifier) + '_created'))

    # if not models:
    #     raise ValueError('train.py: given identifier {} matched no models.'.format(identifier))

    new_model_folder_name = model_dirpath.replace('_created', '_training')
    os_rename(model_dirpath, new_model_folder_name)
    # shutil_move(model_dirpath, new_model_folder_name)
    train_one_model_keras(new_model_folder_name)
    # ks = glob_glob(os_path_join(new_model_folder_name, 'k_*'))
    # for k in ks:
    #     # train_one_model_keras()
    #     # Load model
    #     print('train.py: training {}'.format(k))
    #     model_params_path = os_path_join(k, MODEL_PARAMS_FNAME)
    #     # print('train.py: training model', model_params_path, 'with hyperparams')
    #
    #     # create model
    #     # model, model_params = get_which_model_from_params_fname(LeNet, model_params_path, return_params=True)
    #     # model, model_params = get_which_model_from_params_fname(model_params_path, return_params=True)
    #     # summary(model, (130,))
    #     # configure cuda
    #     if 'cuda' in model_params:
    #         using_cuda = model_params['cuda'] and torch.cuda.is_available()
    #     else:
    #         using_cuda = torch.cuda.is_available()
    #     if using_cuda is True:
    #         # print('train.py: Using device ', torch.cuda.get_device_name(0))
    #         model.cuda()
    #
    #     # save initial weights
    #     if 'save_initial' in model_params and model_params['save_initial'] and model_params['save_dir']:
    #         suffix = '_initial'
    #         path = add_suffix_to_path(MODEL_PARAMS_FNAME['save_dir'], suffix) # pylint: disable=E1126
    #         # print('Saving model weights in : ' + path)
    #         ensure_dir(path)
    #         torch.save(model.state_dict(), os_path_join(path, 'model.dat'))
    #         save_model_params(os.path.join(path, MODEL_PARAMS_FNAME), model_params)
    #
    #     # loss
    #     loss = model_params['cost_function']
    #     if loss not in ['MSE', 'L1', 'SmoothL1']:
    #         raise TypeError('Error must be MSE, L1, or SmoothL1. You gave ' + str(loss))
    #     if loss == 'MSE':
    #         loss = torch.nn.MSELoss()
    #     elif loss == 'L1':
    #         loss = torch.nn.L1Loss()
    #     elif loss == 'SmoothL1':
    #         loss = torch.nn.SmoothL1Loss()
    #
    #     # optimizer
    #     if model_params['optimizer'] == 'Adam':
    #         optimizer = torch.optim.Adam(model.parameters(), lr=model_params['learning_rate'], weight_decay=model_params['weight_decay'])
    #     elif model_params['optimizer'] == 'SGD':
    #         optimizer = torch.optim.SGD(model.parameters(), lr=model_params['learning_rate'], momentum=model_params['momentum'], weight_decay=model_params['weight_decay'])
    #     else:
    #         raise ValueError('model_params[\'optimizer\'] must be either Adam or SGD. Got ' + model_params['optimizer'])
    #
    #
    #     # Load training, validation, and test data
    #     # Load primary training data
    #     num_samples = 10 ** 5
    #
    #     trainer = Trainer(model=model,
    #                       loss=loss,
    #                       optimizer=optimizer,
    #                       patience=model_params['patience'],
    #                       loader_train=loader_train,
    #                       loader_train_eval=loader_train_eval,
    #                       loader_val=loader_val,
    #                       cuda=using_cuda,
    #                       logger=logger,
    #                       data_noise_gaussian=model_params['data_noise_gaussian'],
    #                       save_dir=k)
    #
    #     # run training
    #     trainer.train()

    os_rename(new_model_folder_name, new_model_folder_name.replace('_training', '_trained'))


def main():
    # models_dirname, model_type, model_version
    parser = argparse.ArgumentParser()
    parser.add_argument('how_many', type=int, help='')
    # parser.add_argument('model_type', help='')
    # parser.add_argument('model_version', help='')
    args = parser.parse_args()

    how_many = args.how_many
    create_and_train(how_many)


if __name__ == '__main__':
    # seed_everything()
    main()
