import h5py
import numpy as np
import os
from torch.utils.data import Dataset
import torch


class ApertureDataset(Dataset):
    """Aperture domain dataset."""

    def __init__(self, fname, num_samples, k=4, target_is_data=False):
        """
        Args:
            fname: file name for aperture domain data
            num_samples: number of samples to use from data set
            k: frequency to use
            target_is_data: return data as the target (autoencoder)
        """

        print('dataloader.py: fname =', fname, 'num_samples =', num_samples, 'k =', k, 'target_is_data =', target_is_data)

        self.fname = fname

        # check if files exist
        if not os.path.isfile(fname):
            raise IOError(fname + ' does not exist.')

        # Open file
        f = h5py.File(fname, 'r')

        # Get number of samples available for each type
        real_available = f['/' + str(k) + '/X/real'].shape[0]
        imag_available = f['/' + str(k) + '/X/imag'].shape[0]
        samples_available = min(real_available, imag_available)

        # set num_samples
        if not num_samples:
            num_samples = samples_available

        # make sure num_samples is less than samples_available
        if num_samples > samples_available:
            self.num_samples = samples_available
        else:
            self.num_samples = num_samples

        # load the data
        inputs = np.hstack([f['/' + str(k) + '/X/real'][0:self.num_samples],
                            f['/' + str(k) + '/X/imag'][0:self.num_samples]])
        if target_is_data:
            targets = np.hstack([f['/' + str(k) + '/X/real'][0:self.num_samples],
                                 f['/' + str(k) + '/X/imag'][0:self.num_samples]])
        else:
            targets = np.hstack([f['/' + str(k) + '/Y/real'][0:self.num_samples],
                                 f['/' + str(k) + '/Y/imag'][0:self.num_samples]])

        # convert data to single precision pytorch tensors
        self.data_tensor = torch.from_numpy(inputs).float()
        self.target_tensor = torch.from_numpy(targets).float()

        # close file
        f.close()

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, idx):
        return self.data_tensor[idx], self.target_tensor[idx]
