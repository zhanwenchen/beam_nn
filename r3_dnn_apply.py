# import numpy as np
# import os
# import h5py
# import torch
# from torch import nn
# import argparse
# from scipy.io import savemat
# import time

# modify path
# import sys
# sys.path.insert(0, '../../../../../src')
# from model import FullyConnectedNet
# from utils import read_model_params

import os
import argparse
import time
import h5py
import torch
from torch.cuda import is_available as get_cuda_available
from torch.autograd import Variable
import numpy as np
from scipy.io import savemat

from lib.utils import read_model_params
from lib.lenet import LeNet



if __name__ == "__main__":

    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda', help='Option to use GPU.', action="store_true")
    args = parser.parse_args()

    # cuda flag
    print('torch.cuda.is_available(): ' + str(torch.cuda.is_available()))
    if args.cuda and torch.cuda.is_available():
        print('Using ' + str(torch.cuda.get_device_name(0)))
    else:
        print('Not using CUDA')

    # setup device based on cuda flag
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # load stft data
    f = h5py.File("old_stft.mat", "r")
    stft_real = np.asarray(f['old_stft_real'])
    stft_imag = np.asarray(f['old_stft_imag'])
    N_beams, N_elements, N_segments, N_fft = stft_real.shape
    freqs = np.arange(N_fft)/N_fft

    # combine stft_real and stft_imag
    stft = np.concatenate([stft_real, stft_imag], axis=1)
    del stft_real, stft_imag
    stft_start = stft.copy()

    # move element position axis
    stft = np.moveaxis(stft, 1, 2)

    # reshape the to flatten first two axes
    stft = np.reshape(stft, [N_beams*N_segments, 2*N_elements, N_fft])

    # setup model dirs dictionary
    model_dirs_fname = '../model_dirs.txt'
    if not os.path.isfile(model_dirs_fname):
        model_dirs_fname = '../model_dirs.json'
    f = open(model_dirs_fname, 'r')
    model_dirs = {}
    for line in f:
        [key, value] = line.split(',')
        value = value.rstrip()
        if key.isdigit():
            key = int(key)
        model_dirs[key] = value
    f.close()


    # process stft with networks
    k_mask = list(range(3, 6))
    for k in k_mask:
        print('k: ' + str(k))

        # start timer
        t0 = time.time()

        # load the model
        model_params_fname = os.path.join(model_dirs[k], 'model_params.txt')
        if not os.path.isfile(model_params_fname):
            model_params_fname = os.path.join(model_dirs[k], 'model_params.json')
        model_params = read_model_params(model_params_fname)
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
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(model_params_fname), 'model.dat'), map_location='cpu'))
        model.eval()
        model = model.to(device)

        # select data by frequency
        aperture_data = stft[:, :, k]

        # normalize by L1 norm
        aperture_data_norm = np.linalg.norm(aperture_data, ord=np.inf, axis=1)
        aperture_data = aperture_data / aperture_data_norm[:, np.newaxis]

        # load into torch and onto gpu
        aperture_data = torch.from_numpy(aperture_data).float()
        aperture_data = aperture_data.to(device)

        # process with the network
        with torch.set_grad_enabled(False):
            aperture_data_new = model(aperture_data).to('cpu').data.numpy()

        # rescale the data
        aperture_data_new = aperture_data_new * aperture_data_norm[:, np.newaxis]

        # store new data in stft
        stft[:, :, k] = aperture_data_new

        # delete the model
        del model

        # stop timer
        print('time: {:.2f}'.format(time.time() - t0))

    # reshape the stft data
    stft = np.reshape(stft, [N_beams, N_segments, 2*N_elements, N_fft])

    # set zero outside analysis frequency range
    mask = np.zeros(stft.shape, dtype=np.float32)
    mask[:, :, :, k_mask] = 1
    stft = mask * stft
    del mask

    # mirror data to negative frequencies using conjugate symmetry
    stft[:, :, :, np.int32(N_fft/2)+1:] = np.flip(stft[:, :, :, 1:np.int32(N_fft/2)], axis=3)
    stft[:, :, N_elements:2*N_elements, np.int32(N_fft/2)+1:] = -1 * stft[:, :, N_elements:2*N_elements, np.int32(N_fft/2)+1:]

    # move element position axis
    stft = np.moveaxis(stft, 1, 2)

    # change variable names
    new_stft_real = stft[:, :N_elements, :, :]
    new_stft_imag = stft[:, N_elements:, :, :]

    # change dimensions
    new_stft_real = new_stft_real.transpose()
    new_stft_imag = new_stft_imag.transpose()

    # save new stft data
    savemat('new_stft.mat', {'new_stft_real': new_stft_real, 'new_stft_imag': new_stft_imag})
