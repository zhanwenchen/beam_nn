from torch import from_numpy # pylint: disable=E0611
from torch.utils.data import Dataset


class ApertureDatasetEval(Dataset):
    """Aperture domain dataset."""
    def __init__(self, x):
        """
        Args:
            fname: file name for aperture domain data
            num_samples: number of samples to use from data set
            k: frequency to use
            target_is_data: return data as the target (autoencoder)
        """

        self.x = from_numpy(x).float()

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, index):
        return self.x[index]
