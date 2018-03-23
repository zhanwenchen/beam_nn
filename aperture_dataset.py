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
