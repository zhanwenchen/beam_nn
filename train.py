from os import rename as os_rename
from os.path import join as os_path_join
import argparse
from glob import glob

from torch import save as torch_save
from torch.utils.data import DataLoader
from torch.cuda import is_available as torch_cuda_is_available
from torch import multiprocessing as mp
# from torch.multiprocessing import Pool

from lib.utils import save_model_params, ensure_dir, add_suffix_to_path, get_which_model_from_params_fname
from lib.dataloader import ApertureDataset
from lib.logger import Logger
from lib.trainer import Trainer


model_parent_folder = 'DNNs'
model_params_fname = 'model_params.json'
MODEL_DATA_FNAME = 'model.dat'
NUM_SAMPLES_TRAIN = 10 ** 5
NUM_SAMPLES_TRAIN_EVAL = 10 ** 4
NUM_SAMPLES_VALID = 10 ** 4
DATALOADER_NUM_WORKERS = 4


def train(identifier, num_concurrent_models=-1):
    models = glob(os_path_join(model_parent_folder, str(identifier) + '_created'))

    if not models:
        raise ValueError('train.py: given identifier {} matched no models.'.format(identifier))

    if num_concurrent_models == -1:
        num_concurrent_models = mp.cpu_count()

    with mp.Pool(processes=num_concurrent_models) as pool:
        list(pool.imap_unordered(train_one, models))


def train_one(model_folder):
    new_model_folder_name = model_folder.replace('_created', '_training')
    os_rename(model_folder, new_model_folder_name)
    frequencies = glob(os_path_join(new_model_folder_name, 'k_*'))
    for frequency in frequencies:
        # Load model
        print('train.py: training {}'.format(frequency))
        model_params_path = os_path_join(frequency, model_params_fname)

        # create model
        model, model_params = get_which_model_from_params_fname(model_params_path, return_params=True)
        if 'cuda' in model_params:
            using_cuda = model_params['cuda'] and torch_cuda_is_available()
        else:
            using_cuda = torch_cuda_is_available()

        if using_cuda is True:
            model.cuda()

        # save initial weights
        if 'save_initial' in model_params and model_params['save_initial'] and model_params['save_dir']:
            suffix = '_initial'
            path = add_suffix_to_path(model_params_fname['save_dir'], suffix) # pylint: disable=E1126
            ensure_dir(path)
            torch_save(model.state_dict(), os_path_join(path, MODEL_DATA_FNAME))
            save_model_params(os_path_join(path, model_params_fname), model_params)

        # loss
        if 'cost_function' in model_params:
            loss = model_params['cost_function']
        elif 'loss_function' in model_params:
            loss = model_params['loss_function']
        else:
            raise ValueError('model_params missing key cost_function or loss_function')

        if loss not in ['MSE', 'L1', 'SmoothL1']:
            raise TypeError('Error must be MSE, L1, or SmoothL1. You gave ' + str(loss))
        if loss == 'MSE':
            from torch.nn import MSELoss
            loss = MSELoss()
        elif loss == 'L1':
            from torch.nn import L1Loss
            loss = L1Loss()
        elif loss == 'SmoothL1':
            from torch.nn import SmoothL1Loss
            loss = SmoothL1Loss()

        # optimizer
        if model_params['optimizer'] == 'Adam':
            from torch.optim import Adam
            optimizer = Adam(model.parameters(), lr=model_params['learning_rate'], weight_decay=model_params['weight_decay'])
        elif model_params['optimizer'] == 'SGD':
            from torch.optim import SGD
            optimizer = SGD(model.parameters(), lr=model_params['learning_rate'], momentum=model_params['momentum'], weight_decay=model_params['weight_decay'])
        else:
            raise ValueError('model_params[\'optimizer\'] must be either Adam or SGD. Got ' + model_params['optimizer'])

        logger = Logger()

        # Load training, validation, and test data
        # Load primary training data
        dat_train = ApertureDataset(model_params['data_train'], NUM_SAMPLES_TRAIN, k=model_params['k'], target_is_data=model_params['data_is_target'])
        loader_train = DataLoader(dat_train, batch_size=model_params['batch_size'], shuffle=True, num_workers=DATALOADER_NUM_WORKERS, pin_memory=using_cuda)

        # Load secondary training data - used to evaluate training loss after every epoch
        dat_train2 = ApertureDataset(model_params['data_train'], NUM_SAMPLES_TRAIN_EVAL, k=model_params['k'], target_is_data=model_params['data_is_target'])
        loader_train_eval = DataLoader(dat_train2, batch_size=model_params['batch_size'], shuffle=False, num_workers=DATALOADER_NUM_WORKERS, pin_memory=using_cuda)

        # Load validation data - used to evaluate validation loss after every epoch
        dat_val = ApertureDataset(model_params['data_val'], NUM_SAMPLES_VALID, k=model_params['k'], target_is_data=model_params['data_is_target'])
        loader_val = DataLoader(dat_val, batch_size=model_params['batch_size'], shuffle=False, num_workers=DATALOADER_NUM_WORKERS, pin_memory=using_cuda)

        trainer = Trainer(model=model,
                          loss=loss,
                          optimizer=optimizer,
                          patience=model_params['patience'],
                          loader_train=loader_train,
                          loader_train_eval=loader_train_eval,
                          loader_val=loader_val,
                          cuda=using_cuda,
                          logger=logger,
                          data_noise_gaussian=model_params['data_noise_gaussian'],
                          save_dir=frequency)

        # run training
        trainer.train()

    os_rename(new_model_folder_name, new_model_folder_name.replace('_training', '_trained'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('identifier', help='Option to load model params from a file. Values in this file take precedence.')
    parser.add_argument('num_concurrent_models', type=int, help='Number of models to train at the same time')
    args = parser.parse_args()

    # seed_everything()
    identifier = args.identifier
    num_concurrent_models = args.num_concurrent_models
    train(identifier, num_concurrent_models=num_concurrent_models)


if __name__ == '__main__':
    main()
