from torch import is_tensor as torch_is_tensor
from torch import ones as torch_ones # pylint: disable=E0611
from torch import float64 as torch_float64 # pylint: disable=E0611
from torchaudio.functional import istft as torch_istft


def istft_wrapper(stft_object):
    stft_data = stft_object['stft']
    assert torch_is_tensor(stft_data)
    shift_length = stft_object['shift_length']
    y, _, _ = stft_object['origSigSize']
    len_each_section, _, _, _ = stft_data.shape
    window = torch_ones(len_each_section, dtype=torch_float64)
    return torch_istft(stft_data, len_each_section, window=window,
                       hop_length=shift_length, center=False, onesided=False,
                       normalized=False, pad_mode='constant', length=y)
