#!/usr/bin/env python

from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch import nn, optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch
import os
import h5py
import numpy as np
import warnings
import time
from math import cos, pi
import argparse


class ApertureDataset(Dataset):
    """Aperture domain dataset."""

    # REVIEW: k=4 bad?
    def __init__(self, fname, num_samples, k=4):
        """
        Args:
            fname: file name for aperture domain data
            num_samples: number of samples to use from data set
            k: frequency to use
        """

        self.fname = fname
        self.num_samples = num_samples

        # check if files exist
        if not os.path.isfile(fname):
            raise IOError(fname + ' does not exist.')

        # Open file
        f = h5py.File(fname, 'r')

        # Get number of samples available for each type
        real_available = f['/' + str(k) + '/aperture_data/real'].shape[0]
        imag_available = f['/' + str(k) + '/aperture_data/imag'].shape[0]
        samples_available = min(real_available, imag_available)

        # set num_samples
        if not num_samples:
            num_samples = samples_available

        # make sure num_samples is less than samples_available
        if num_samples > samples_available:
            warnings.warn('data_size > self.samples_available. Setting data_size to samples_available')
            self.num_samples = self.samples_available
        else:
            self.num_samples = num_samples

        # load the data
        inputs = np.hstack([ f['/' + str(k) + '/aperture_data/real'][0:self.num_samples],
                            f['/' + str(k) + '/aperture_data/imag'][0:self.num_samples] ] )
        targets = np.hstack([ f['/' + str(k) + '/targets/real'][0:self.num_samples],
                            f['/' + str(k) + '/targets/imag'][0:self.num_samples] ] )

        # normalize the training data
        C = np.max(np.abs(inputs), axis=1)[:, np.newaxis]
        C[np.where(C==0)[0]] = 1
        inputs = inputs / C
        targets = targets / C

        # convert data to single precision pytorch tensors
        self.data_tensor = torch.from_numpy(inputs).float()
        self.target_tensor = torch.from_numpy(targets).float()

        # close file
        f.close()

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, idx):
        return self.data_tensor[idx], self.target_tensor[idx]





class FullyConnectedNet(nn.Module):
    """Fully connected network with. There are five hidden layers.
        ReLU is the activation function. Network parameters are intialized
        with a normal distribution.

    Args:
        input_dim
        output_dim
        layer_width

    """

    # self.features = nn.Sequential(
    #         nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(kernel_size=3, stride=2),
    #         nn.Conv2d(64, 192, kernel_size=5, padding=2),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(kernel_size=3, stride=2),
    #         nn.Conv2d(192, 384, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(384, 256, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(256, 256, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(kernel_size=3, stride=2),
    #     )
    #     self.classifier = nn.Sequential(
    #         nn.Dropout(),
    #         nn.Linear(256 * 6 * 6, 4096),
    #         nn.ReLU(inplace=True),
    #         nn.Dropout(),
    #         nn.Linear(4096, 4096),
    #         nn.ReLU(inplace=True),
    #         nn.Linear(4096, num_classes),
    #     )

    # def __init__(self, input_dim, output_dim, layer_width):
    #     #super(FullyConnectedNet, self).__init__()
    #     super().__init__()
    #
    #     self.input_layer = nn.Linear(input_dim, layer_width)
    #     self.hidden_1 = nn.Linear(layer_width, layer_width)
    #     self.hidden_2 = nn.Linear(layer_width, layer_width)
    #     self.hidden_3 = nn.Linear(layer_width, layer_width)
    #     self.hidden_4 = nn.Linear(layer_width, layer_width)
    #     self.hidden_5 = nn.Linear(layer_width, layer_width)
    #     self.output_layer = nn.Linear(layer_width, output_dim)
    #     self.relu = nn.ReLU()
    #
    #     self._initialize_weights()

    def __init__(self, input_dim, output_dim, layer_width, num_hidden_layers):
        # super(MyNet, self).__init__()
        super().__init__()

        self.layers = []
        self.layers.append(nn.Linear(input_dim, layer_width))
        for i in range(num_hidden_layers):
            self.layers.append(nn.Linear(layer_width, layer_width))
        self.layers.append(nn.Linear(layer_width, output_dim))

        self.relu = nn.ReLU()
        self._initialize_weights()

    def forward(self, x):
        x = None
        for layer in self.layers:
            x = self.relu(layer(x))
        # x = self.relu( self.input_layer(x) )
        # x = self.relu(self.hidden_1(x) )
        # x = self.relu(self.hidden_2(x) )
        # x = self.relu(self.hidden_3(x) )
        # x = self.relu(self.hidden_4(x) )
        # x = self.relu(self.hidden_5(x) )
        # x = self.output_layer(x)

        return x

    def _initialize_weights(self):

        for layer in self.layers:
            nn.init.kaiming_normal(layer.weight.data )
            layer.bias.data.fill_(0)

        # nn.init.kaiming_normal( self.input_layer.weight.data )
        # self.input_layer.bias.data.fill_(0)
        #
        # nn.init.kaiming_normal( self.hidden_1.weight.data )
        # self.hidden_1.bias.data.fill_(0)
        #
        # nn.init.kaiming_normal( self.hidden_2.weight.data )
        # self.hidden_2.bias.data.fill_(0)
        #
        # nn.init.kaiming_normal( self.hidden_3.weight.data )
        # self.hidden_3.bias.data.fill_(0)
        #
        # nn.init.kaiming_normal( self.hidden_4.weight.data )
        # self.hidden_4.bias.data.fill_(0)
        #
        # nn.init.kaiming_normal( self.hidden_5.weight.data )
        # self.hidden_5.bias.data.fill_(0)
        #
        # nn.init.kaiming_normal( self.output_layer.weight.data )
        # self.output_layer.bias.data.fill_(0)


def loss_compute(model, dat_loader, loss_fn):
    model.eval()

    loss = 0
    for i, data in enumerate(dat_loader):
        inputs = Variable(data[0], requires_grad=False)
        targets = Variable(data[1], requires_grad=False)
        if args.cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        outputs = model(inputs)

        loss = loss + loss_fn(outputs, targets).data[0]

    return loss / len(dat_loader)


def train():
    model.train()

    # keep track of loss per batch
    loss_per_batch = []
    lr_per_batch = []

    for batch_idx, data in enumerate(train_loader):

        lr_per_batch.append( optimizer.state_dict()['param_groups'][0]['lr'] )

        inputs = Variable(data[0], requires_grad=False)
        targets = Variable(data[1], requires_grad=False)
        if args.cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        outputs = model(inputs)

        model.zero_grad()

        loss = loss_fn(outputs, targets)

        loss_per_batch.append(loss.data[0])

        loss.backward()

        optimizer.step()

    return loss_per_batch, lr_per_batch


def read_model_params(fname):
    f = open(fname, 'r')
    model_param_dict = {}
    for line in f:
        [key, value] = line.split(',')
        value = value.rstrip()
        if value.isdigit():
            value = int(value)
        model_param_dict[key] = value
    f.close()
    return model_param_dict

def save_model_params(fname, model_params_dict):
    f = open(fname, 'w')
    for key, value in model_params_dict.items():
        print(','.join([str(key), str(value)]), file=f)
    f.close()





if __name__ == '__main__':

    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('fname_train', help='Filename with path to training data.')
    parser.add_argument('fname_validate', help='Filename with path to validation data.')
    parser.add_argument('save_dir', help='Directory to save the model.')
    parser.add_argument('k', help='Integer value. DFT frequency to analyze.', type=int)
    parser.add_argument('-c', '--cuda', help='Option to use GPU.', action="store_true")
    args = parser.parse_args()

    # Load primary training data
    num_samples = 10 ** 5
    dat_train = ApertureDataset(args.fname_train, num_samples)
    train_loader = DataLoader(dat_train, batch_size=1000, shuffle=True, num_workers=1)

    # Load secondary training data - used to evaluate training loss after every epoch
    num_samples = 10 ** 4
    dat_train2 = ApertureDataset(args.fname_train, num_samples)
    train_loader2 = DataLoader(dat_train2, batch_size=1000, shuffle=False, num_workers=1)

    # Load validation data - used to evaluate validation loss after every epoch
    num_samples = 10 ** 4
    dat_validate = ApertureDataset(args.fname_validate, num_samples)
    validate_loader = DataLoader(dat_validate, batch_size=1000, shuffle=False, num_workers=1)

    # cuda flag
    print('torch.cuda.is_available(): ' + str(torch.cuda.is_available()))
    if args.cuda and torch.cuda.is_available():
        print('Using ' + str(torch.cuda.get_device_name(0)))
    else:
        print('Not using CUDA')

    # create model params
    model_params_dict = {}
    model_params_dict['input_dim'] = 130 # TODO move? TODO why move this?
    model_params_dict['output_dim'] = 130
    model_params_dict['layer_width'] = 260 # TODO move?
    model_params_dict['num_hidden_layers'] = 5 # TODO move?
    save_model_params(os.path.join(args.save_dir, 'model_params.txt'), model_params_dict)

    # create model
    model = FullyConnectedNet(**model_params_dict)
    print('\n\nmodel_params_dict =', model_params_dict, '\n\n')
    print('\n\nmodel.parameters() =', model.parameters, '\n\n')
    if args.cuda:
        model.cuda()

    # loss
    loss_fn = nn.MSELoss()
    if args.cuda:
        loss_fn = loss_fn.cuda()

    # optimizer
    lr = 5e-1
    momentum = 0
    # optimizer = optim.SGD(model.parameters(), lr, momentum)
    optimizer = optim.SGD([param.parameters() for param in model.layers], lr, momentum)

    # setup metric recording
    loss_train_batch_history = []
    lr_batch_history = []
    loss_train_epoch_history = []
    loss_valid_epoch_history = []
    time_epoch_history = []
    epoch_list = []

    # setup initial loss_valid
    epoch = 1
    loss_valid_best = 100
    patience = 1
    num_epochs_increased = 0
    best_epoch = 1

    # Perform training
    while True:

        # Run one iteration of SGD
        t0 = time.time()
        loss_per_batch, lr_per_batch = train()
        loss_train_batch_history = loss_train_batch_history + loss_per_batch
        lr_batch_history = lr_batch_history + lr_per_batch

        # Estimate training and validation losses
        loss_train = loss_compute(model, train_loader2, loss_fn)
        loss_valid = loss_compute(model, validate_loader, loss_fn)
        loss_train_epoch_history.append(loss_train)
        loss_valid_epoch_history.append(loss_valid)
        d_loss = (loss_valid-loss_valid_best)/loss_valid_best * 100
        time_epoch = time.time() - t0
        time_epoch_history.append(time_epoch)
        epoch_list.append(epoch)

        # display results
        print('E: {:} / Train: {:.3e} / Valid: {:.3e} / Diff Valid: {:.2f}% / Diff Valid-Train: {:.1f}% / Time: {:.2f}'.format(epoch, loss_train, loss_valid, d_loss, (loss_valid - loss_train)/loss_train*100, time_epoch))

        # if validation loss improves
        if d_loss < 0:
            num_epochs_increased = 0

            # record epoch and loss
            best_epoch = epoch
            loss_valid_best = loss_valid

            # save the model
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'model.dat'))

            # save the other stuff
            np.savez(os.path.join(args.save_dir, 'loss_results'),
                        loss_train_batch_history=loss_train_batch_history,
                        lr_batch_history=lr_batch_history,
                        loss_train_epoch_history=loss_train_epoch_history,
                        loss_valid_epoch_history=loss_valid_epoch_history,
                        time_epoch_history=time_epoch_history,
                        epoch_list=epoch_list,
                        best_epoch=epoch)

        else:
            num_epochs_increased = num_epochs_increased + 1

        # stop training if we lose patience:
        if num_epochs_increased > patience:
            break

        # advance epoch counter
        epoch = epoch + 1

    # save the other stuff
    np.savez(os.path.join(args.save_dir, 'loss_results'),
                loss_train_batch_history=loss_train_batch_history,
                lr_batch_history=lr_batch_history,
                loss_train_epoch_history=loss_train_epoch_history,
                loss_valid_epoch_history=loss_valid_epoch_history,
                time_epoch_history=time_epoch_history,
                epoch_list=epoch_list,
                best_epoch=epoch)
