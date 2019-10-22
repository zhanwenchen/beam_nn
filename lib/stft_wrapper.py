from math import ceil as math_ceil
from logging import getLogger as logging_getLogger

from torch import stft as torch_stft
from torch import arange as torch_arange # pylint: disable=E0611
from torch import ones as torch_ones, float64 as torch_float64 # pylint: disable=E0611
from torch import is_tensor as torch_is_tensor
from torch import from_numpy as torch_from_numpy # pylint: disable=E0611

# from logging import debug as logging_debug
LOGGER = logging_getLogger('evaluate_keras')

def stft(signal, len_each_section, frac_overlap, padding, window=None):
    try:
        assert torch_is_tensor(signal)
    except:
        signal = torch_from_numpy(signal).double()
    if signal.is_contiguous() is False:
        LOGGER.debug('stft: signal is not contiguous')
        signal = signal.contiguous()
    if window is None:
        window = torch_ones(len_each_section, dtype=torch_float64)
    else:
        raise NotImplementedError('stft: window function {} has not been implemented'.format(window))
    shift_length = round(len_each_section * (1. - frac_overlap)) # shift_length = 2

    y, num_elements, num_beams = signal.shape

    num_frames = math_ceil((y - len_each_section + 1) / shift_length)

    startLocs = torch_arange(0, num_frames*shift_length, shift_length)

    num_elements_beams = num_elements*num_beams
    freq = torch_arange(padding)/padding
    signal_stft = torch_stft(signal.view(y, num_elements_beams).permute(1, 0),
                             len_each_section, window=window,
                             hop_length=shift_length, center=False,
                             onesided=False, normalized=False,
                             pad_mode='constant') \
                             .permute(1, 2, 0, 3) \
                             .view(len_each_section, num_frames, num_elements, num_beams, 2)

    return {
        'stft': signal_stft,
        'freqs': freq,
        'startOffsets': startLocs,
        'len_each_section': len_each_section,
        'padding': padding,
        'win_info': window,
        'frac_overlap': frac_overlap,
        'shift_length': shift_length,
    }
