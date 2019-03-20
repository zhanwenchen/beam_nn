import os
# import argparse
# import time
import h5py
# import glob

# from torchsummary import summary

import numpy as np
from scipy.io import savemat
import torch
from torch import from_numpy, device
from torch.cuda import is_available as get_cuda_available

from lib.utils import get_which_model_from_params_fname
from lib.lenet import LeNet


SCRIPT_FNAME = os.path.basename(__file__)
MODEL_PARAMS_FNAME = 'model_params.json'



def main(target_dirname, cuda=True):
    scan_battery_dirname = os.path.dirname(target_dirname)
    model_dirname = os.path.dirname(os.path.dirname(scan_battery_dirname))

    is_using_cuda = cuda and get_cuda_available()

    # cuda flag
    # if is_using_cuda is True:
    #     print('r3_dnn_apply.py: Using ', torch.cuda.get_device_name(0))
    # else:
    #     print('r3_dnn_apply.py: Not using CUDA')

    # setup device based on cuda flag
    my_device = device('cuda:0' if is_using_cuda else 'cpu')

    # load stft data
    old_stft_fname = os.path.join(target_dirname, 'old_stft.mat')
    with h5py.File(old_stft_fname, 'r') as f:
        stft_real = np.asarray(f['old_stft_real'])
        stft_imag = np.asarray(f['old_stft_imag'])
    N_beams, N_elements, N_segments, N_fft = stft_real.shape
    # freqs = np.arange(N_fft)/N_fft

    # combine stft_real and stft_imag
    stft = np.concatenate([stft_real, stft_imag], axis=1)
    del stft_real, stft_imag

    # move element position axis
    stft = np.moveaxis(stft, 1, 2) # TODO: Duplicate?

    # reshape the to flatten first two axes
    stft = np.reshape(stft, [N_beams*N_segments, 2*N_elements, N_fft]) # TODO: Duplicate?

    # setup model dirs dictionary
    # model_dirs_fname = '../model_dirs.txt'
    # model_dirs_fname = os.path.join(target_dirname, '..', 'model_dirs.txt')
    # if not os.path.isfile(model_dirs_fname):
    #     # model_dirs_fname = '../model_dirs.json'
    #     model_dirs_fname = os.path.join(target_dirname, '..', '/model_dirs.json')
    #     # model_dirs_fname = '../model_dirs.json'

    # model_dirs = {}
    # with open(model_dirs_fname, 'r') as f:
    #     for line in f:
    #         [key, value] = line.split(',')
    #         value = value.rstrip()
    #         if key.isdigit():
    #             key = int(key)
    #         model_dirs[key] = value


    # process stft with networks
    # ks = glob.glob(os.path.join(model_dirname, 'k_*'))
    k_mask = list(range(3, 6))
    for k in k_mask:
        # print('r3_dnn_apply: k =', k)

        # start timer
        # t0 = time.time()

        # load the model
        # model_params_fname = os.path.join(model_dirs[k], 'model_params.txt')
        # if not os.path.isfile(model_params_fname):
            # model_params_fname = os.path.join(model_dirs[k], 'model_params.json')
        model_params_fname = os.path.join(os.path.join(model_dirname, 'k_' + str(k)), MODEL_PARAMS_FNAME)
        # print('{}: model_params_fname = {}'.format(SCRIPT_FNAME, model_params_fname))
        try
        model = get_which_model_from_params_fname(LeNet, model_params_fname)


        model.load_state_dict(torch.load(os.path.join(os.path.dirname(model_params_fname), 'model.dat'), map_location='cpu'))
        model.eval()
        model = model.to(my_device)

        # select data by frequency
        aperture_data = stft[:, :, k]

        # normalize by L1 norm
        aperture_data_norm = np.linalg.norm(aperture_data, ord=np.inf, axis=1)
        aperture_data = aperture_data / aperture_data_norm[:, np.newaxis]

        # load into torch and onto gpu
        aperture_data = from_numpy(aperture_data).float()
        aperture_data = aperture_data.to(my_device)

        # process with the network
        with torch.set_grad_enabled(False):
            aperture_data_new = model(aperture_data).to('cpu').data.numpy()

        # summary(model, aperture_data.shape)
        # delete the model
        del model

        # rescale the data and store new data in stft
        stft[:, :, k] = aperture_data_new * aperture_data_norm[:, np.newaxis]

        # stop timer
        # print('time: {:.2f}'.format(time.time() - t0))

    # reshape the stft data
    stft = np.reshape(stft, [N_beams, N_segments, 2*N_elements, N_fft]) # TODO: Duplicate?

    # set zero outside analysis frequency range
    mask = np.zeros(stft.shape, dtype=np.float32)
    mask[:, :, :, k_mask] = 1
    stft = mask * stft
    del mask

    # mirror data to negative frequencies using conjugate symmetry
    stft[:, :, :, np.int32(N_fft/2)+1:] = np.flip(stft[:, :, :, 1:np.int32(N_fft/2)], axis=3)
    stft[:, :, N_elements:2*N_elements, np.int32(N_fft/2)+1:] = -1 * stft[:, :, N_elements:2*N_elements, np.int32(N_fft/2)+1:]

    # move element position axis
    stft = np.moveaxis(stft, 1, 2) # TODO: Duplicate?

    # change variable names
    new_stft_real = stft[:, :N_elements, :, :]
    new_stft_imag = stft[:, N_elements:, :, :]

    # change dimensions
    new_stft_real = new_stft_real.transpose()
    new_stft_imag = new_stft_imag.transpose()

    # save new stft data
    new_stft_fname = os.path.join(target_dirname, 'new_stft.mat')
    savemat(new_stft_fname, {'new_stft_real': new_stft_real, 'new_stft_imag': new_stft_imag})

# if __name__ == "__main__":
#     main()
