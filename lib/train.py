#!/usr/bin/env python
import torch
import os
import numpy as np
import time
import argparse
import glob

import warnings
from pprint import pprint


from utils import read_model_params, save_model_params, ensure_dir, add_suffix_to_path
from dataloader import ApertureDataset
from lenet import LeNet
from logger import Logger
from trainer import Trainer


def train(identifier):
    models = glob.glob(os.path.join('DNNs', str(identifier) + '*'))

    for model in models:
        ks = glob.glob(os.path.join(model, 'k_*'))
        for k in ks:
            model_params_path = k + '/model_params.txt'
            print('train.py: training model', model_params_path, 'with hyperparams')

            # load model params if it is specified
            model_params = read_model_params(model_params_path)

            # display input arguments
            pprint(model_params)

            # cuda flag
            # print('torch.cuda.is_available(): ' + str(torch.cuda.is_available()))
            using_cuda = model_params['cuda'] and torch.cuda.is_available()
            if using_cuda:
                print('Using ' + str(torch.cuda.get_device_name(0)))
            else:
                print('Not using CUDA')
                warnings.warn('Not using CUDA')

            # Load primary training data
            num_samples = 10 ** 5
            dat_train = ApertureDataset(model_params['data_train'], num_samples, model_params['k'], model_params['data_is_target'])
            loader_train = torch.utils.data.DataLoader(dat_train, batch_size=model_params['batch_size'], shuffle=True, num_workers=1)

            # Load secondary training data - used to evaluate training loss after every epoch
            num_samples = 10 ** 4
            dat_train2 = ApertureDataset(model_params['data_train'], num_samples, model_params['k'], model_params['data_is_target'])
            loader_train_eval = torch.utils.data.DataLoader(dat_train2, batch_size=model_params['batch_size'], shuffle=False, num_workers=1)

            # Load validation data - used to evaluate validation loss after every epoch
            num_samples = 10 ** 4
            dat_val = ApertureDataset(model_params['data_val'], num_samples, model_params['k'], model_params['data_is_target'])
            loader_val = torch.utils.data.DataLoader(dat_val, batch_size=model_params['batch_size'], shuffle=False, num_workers=1)

            # create model
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


            if using_cuda:
                model.cuda()

            # save initial weights
            if model_params['save_initial'] and model_params['save_dir']:
                suffix = '_initial'
                path = add_suffix_to_path(model_parmas['save_dir'], suffix)
                print('Saving model weights in : ' + path)
                ensure_dir(path)
                torch.save(model.state_dict(), os.path.join(path, 'model.dat'))
                save_model_params(os.path.join(path, 'model_params.txt'), model_params)

            # loss
            loss = torch.nn.MSELoss()

            # optimizer
            if model_params['optimizer'] == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=model_params['learning_rate'], weight_decay=model_params['weight_decay'])
            elif model_params['optimizer'] == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(), lr=model_params['learning_rate'], momentum=model_params['momentum'], weight_decay=model_params['weight_decay'])
            else:
                raise ValueError('model_params[\'optimizer\'] must be either Adam or SGD. Got ' + model_params['optimizer'])


            # logger
            logger = Logger()

            # trainer
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
                                save_dir=model_params['save_dir'])

            # run training
            trainer.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('identifier', help='Option to load model params from a file. Values in this file take precedence.')
    args = parser.parse_args()

    identifier = args.identifier
    train(identifier)


if __name__ == '__main__':
    main()
