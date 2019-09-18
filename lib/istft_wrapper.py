from torch import is_tensor as torch_is_tensor
from torch import ones as torch_ones # pylint: disable=E0611
from torch import float64 as torch_float64 # pylint: disable=E0611
from torchaudio.functional import istft as torch_istft


def istft(stft_object):
    stft_data = stft_object['stft']
    assert torch_is_tensor(stft_data)
    len_each_section, num_frames, num_elements, num_beams, real_imag = stft_data.size()
    num_elements_beams = num_elements*num_beams
    shift_length = stft_object['shift_length']
    y, _, _ = stft_object['origSigSize']
    len_each_section = stft_data.size(0)
    window = torch_ones(len_each_section, dtype=torch_float64)
    return torch_istft(stft_data.view(len_each_section, num_frames, \
                       num_elements_beams, real_imag).permute(2, 0, 1, 3),
                       len_each_section, window=window,
                       hop_length=shift_length, center=False, onesided=False,
                       normalized=False, pad_mode='constant', length=y) \
                       .view(num_elements, num_beams, y).permute(2, 0, 1)
