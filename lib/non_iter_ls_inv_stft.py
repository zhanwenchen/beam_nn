from math import sqrt as math_sqrt

from numpy import zeros as np_zeros, arange as np_arange, newaxis as np_newaxis
from numpy.fft import ifft as np_ifft

from lib.stft import stft


def non_iter_ls_inv_stft(stft_object):
    stft_data = stft_object['stft']
    origSigSize = stft_object['origSigSize']
    num_rows, _, _ = origSigSize
    shift_length = stft_object['shift_length']
    len_each_section, num_rows_overlap, _, _ = stft_data.shape
    # TODO: Isn't this just num_rows in the very beginning?
    # total_new_elements = (num_rows_overlap - 1) * shift_length + len_each_section
    win_info = stft_object['win_info']
    wVec = win_info(len_each_section)
    wVecSq = wVec ** 2
    vecC = np_arange(1, num_rows_overlap*shift_length, step=shift_length)
    # vecC = range(0, num_rows_overlap*shift_length-1, shift_length)
    DlsArr = np_zeros((num_rows,))
    for j in vecC:
        tmpArr = np_arange(j-1, len_each_section+j-1)
        # tmpArr = np_arange(j, len_each_section+j)
        DlsArr[tmpArr] += wVecSq
    # DlsArrInv = 1/DlsArr
    invFT = math_sqrt(len_each_section)*np_ifft(stft_data, axis=0)
    invFT_real = invFT.real
    invFT *= wVec[:, np_newaxis, np_newaxis, np_newaxis]
    yEst = np_zeros(origSigSize)
    for index, j in enumerate(vecC):
        tmpArr = np_arange(j-1, len_each_section+j-1)
        yEst[tmpArr, :] += invFT_real[:, index, :]
    # sigOut = yEst * DlsArrInv[:, np_newaxis, np_newaxis]
    sigOut = yEst / DlsArr[:, np_newaxis, np_newaxis]
    return sigOut
