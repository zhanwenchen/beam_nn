"""
Run r6_dnn_image_display.mat first to generate and save masks.mat and then
use this load_and_save_masks to convert it from matlab (masks.mat) to numpy (masks.npz)
"""
from os.path import join as os_path_join

from scipy.io import loadmat
from numpy import savez as np_savez


MASKS_LOAD_FNAME_MATLAB = 'masks.mat'
MASKS_SAVE_FNAME_NUMPY = 'masks' # np_savez expands this to masks.npz by default.

def load_and_save_masks(target_dirname):
    # In Matlab, need to do `save('masks.mat', 'mask_in', 'mask_out')`
    masks_mat = loadmat(os_path_join(target_dirname, MASKS_LOAD_FNAME_MATLAB))
    # masks_mat = loadmat('masks.mat')
    mask_in_matlab = masks_mat['mask_in']
    mask_out_matlab = masks_mat['mask_out']
    mask_in = mask_in_matlab.astype(bool)
    mask_out = mask_out_matlab.astype(bool)
    np_savez(os_path_join(target_dirname, MASKS_SAVE_FNAME_NUMPY), mask_in=mask_in, mask_out=mask_out)
