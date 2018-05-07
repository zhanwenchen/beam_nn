import os
import h5py
import numpy as np
import warnings
import torch
from torch.utils.data import Dataset

class ApertureDataset(Dataset):
    """Aperture domain dataset."""

    # REVIEW: k=4 bad?
    def __init__(self, fname, num_samples, k):
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
        x_real = np.array(f['/' + str(k) + '/aperture_data/real'][0:self.num_samples])
        x_imaginery = np.array(f['/' + str(k) + '/aperture_data/imag'][0:self.num_samples])

        y_real = np.array(f['/' + str(k) + '/targets/real'][0:self.num_samples])
        y_imaginery = np.array(f['/' + str(k) + '/targets/imag'][0:self.num_samples])


        # Normalization
        # REVIEW: C is calculate only from inputs. WHy not both inputs and targets?
        C_real = np.max(np.abs(x_real))
        if C_real == 0: C_real = 1
        C_imaginery = np.max(np.abs(x_imaginery))
        if C_imaginery == 0: C_imaginery = 1

        x_real, y_real = x_real / C_real, y_real / C_real
        x_imaginery, y_imaginery = x_imaginery / C_real, y_imaginery / C_real


        # Stacking x as (2 x 65) and y as (1 x 130)
        x = np.stack((x_real, x_imaginery), axis=1)
        y = np.hstack((y_real, y_imaginery))


        # Convert data to single precision pytorch tensors.
        self.data_tensor = torch.from_numpy(x).float() # REVIEW: cuda()?
        self.target_tensor = torch.from_numpy(y).float()

        f.close()

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, idx):
        return self.data_tensor[idx], self.target_tensor[idx]
