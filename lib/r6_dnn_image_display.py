from os.path import join as os_path_join
from math import log10 as math_log10, sqrt as math_sqrt
from json import dump as json_dump
from logging import info as logging_info, \
                    INFO as logging_INFO, \
                    debug as loggin_debug

from scipy.io import loadmat
# from numpy import meshgrid as np_meshgrid
# from numpy import sqrt as np_sqrt
from numpy import squeeze as np_squeeze, load as np_load
from matplotlib import use as matplotlib_use
matplotlib_use('Agg') # NOTE: Important: this prevents plt from blocking rest of code
from matplotlib.pyplot import subplots as plt_subplots, \
                              close as plt_close
# from lib.utils import load_single_value


DNN_IMAGE_FNAME = 'dnn_image.mat'
FONT_SIZE = 20
PROCESS_SCRIPTS_DIRNAME = 'process_scripts'
CIRCLE_RADIUS_FNAME = 'circle_radius.txt'
CIRCLE_COORDS_X_FNAME = 'circle_xc.txt'
CIRCLE_COORDS_Y_FNAME = 'circle_zc.txt'
BOX_XMIN_RIGHT_FNAME = 'box_right_min.txt'
BOX_XMAX_RIGHT_FNAME = 'box_right_max.txt'
BOX_XMIN_LEFT_FNAME = 'box_left_min.txt'
BOX_XMAX_LEFT_FNAME = 'box_left_max.txt'
SPECKLE_STATS_FNAME = 'speckle_stats_dnn.txt'
SPECKLE_STATS_DICT_FNAME = 'speckle_stats_dnn.json'
DNN_IMAGE_SAVE_FNAME = 'dnn.png'
MASKS_FNAME = 'masks.npz'


def r6_dnn_image_display(target_dirname, dnn_image_obj=None, show_fig=False):
    logging_info('{}: r6: Turning upsampled envelope into image...'.format(target_dirname))
    if dnn_image_obj is None:
        dnn_image_obj = loadmat(os_path_join(target_dirname, DNN_IMAGE_FNAME))

    beam_position_x_up = dnn_image_obj['beam_position_x_up']
    depth = dnn_image_obj['depth']
    envUp_dB = dnn_image_obj['envUp_dB']
    env_up = dnn_image_obj['envUp']

    loggin_debug('{}: r6: Finished loading vars'.format(target_dirname))
    x = np_squeeze(beam_position_x_up) # beam_position_x_up
    y = np_squeeze(depth) # depth

    loggin_debug('{}: r6: Finished squeezing x, y'.format(target_dirname))
    fig, ax = plt_subplots()
    loggin_debug('{}: r6: Finished plt.figure'.format(target_dirname))
    image = ax.imshow(envUp_dB, vmin=-60, vmax=0, cmap='gray', aspect='auto', extent=[x[0]*1000, x[-1]*1000, y[-1]*1000, y[0]*1000])
    ax.set_aspect('equal')
    loggin_debug('{}: r6: Finished plt.imshow'.format(target_dirname))
    fig.colorbar(image)
    loggin_debug('{}: r6: Finished plt.colorbar'.format(target_dirname))
    # plt_xlabel('lateral (mm)', fontsize=FONT_SIZE)
    ax.set_xlabel('lateral (mm)', fontsize=FONT_SIZE)
    # plt_ylabel('axial (mm)', fontsize=FONT_SIZE)
    ax.set_ylabel('axial (mm)', fontsize=FONT_SIZE)
    loggin_debug('{}: r6: Finished plt.xlabel/ylabel'.format(target_dirname))

    # if show_fig is True:
    # plt_show(block=False)

    # Save image to file
    dnn_image_path = os_path_join(target_dirname, DNN_IMAGE_SAVE_FNAME)
    fig.savefig(dnn_image_path)
    plt_close(fig)

    loggin_debug('{}: r6: Finished saving figure'.format(target_dirname))
    # scan_battery_dirname = os_path_dirname(target_dirname)
    # process_scripts_dirpath = os_path_join(scan_battery_dirname, PROCESS_SCRIPTS_DIRNAME)

    # circle_radius = load_single_value(process_scripts_dirpath, CIRCLE_RADIUS_FNAME)
    # circle_coords_x = load_single_value(process_scripts_dirpath, CIRCLE_COORDS_X_FNAME)
    # circle_coords_y = load_single_value(process_scripts_dirpath, CIRCLE_COORDS_Y_FNAME)

    # xx, yy = np_meshgrid(x, y)
    # mask_in = get_circular_mask(xx, yy, (circle_coords_x, circle_coords_y), circle_radius)
    #
    # # mask_in
    # # create rectangular region outside lesion
    # box_xmin_right = load_single_value(process_scripts_dirpath, BOX_XMIN_RIGHT_FNAME)
    # box_xmax_right = load_single_value(process_scripts_dirpath, BOX_XMAX_RIGHT_FNAME)
    #
    # box_xmin_left = load_single_value(process_scripts_dirpath, BOX_XMIN_LEFT_FNAME)
    # box_xmax_left = load_single_value(process_scripts_dirpath, BOX_XMAX_LEFT_FNAME)
    #
    # # Box shares y position and height with circle (diameter)
    # ymin = circle_coords_y - circle_radius
    # ymax = circle_coords_y + circle_radius
    # mask_out_left = (xx >= box_xmin_left) * (xx <= box_xmax_left) * (yy >= ymin) * (yy <= ymax)
    # mask_out_right = get_rectangle_mask(xx, yy, box_xmin_right, box_xmax_right, ymin, ymax)
    # mask_out = mask_out_left | mask_out_right

    # Display circle and boxes
    # with_circle = envUp_dB.copy()
    #
    #
    # with_circle[mask_out_left+mask_in+mask_out_right] = 0
    #
    # plt.figure(figsize=(12,16))
    # plt.imshow(with_circle, vmin=-60, vmax=0, cmap='gray', aspect='auto', extent = [beam_position_x_up[0]*1000,beam_position_x_up[-1]*1000,depth[-1]*1000, depth[0]*1000])
    # plt.colorbar()
    # FONT_SIZE = 20
    # plt.xlabel('lateral (mm)', fontsize=FONT_SIZE)
    # plt.ylabel('axial (mm)', fontsize=FONT_SIZE)

    # plt.show()

    # Calculate image statistics
    # print('r6: env_up.shape =', env_up.shape)
    mask_in, mask_out = get_masks(target_dirname)
    loggin_debug('{}: r6: Finished loading masks'.format(target_dirname))
    # print('r6: mask_in.shape={}, mask_out.shape={}'.format(mask_in.shape, mask_out.shape))
    env_up_inside_lesion = env_up[mask_in]
    mean_in = env_up_inside_lesion.mean()
    var_in = env_up_inside_lesion.var(ddof=1) # ddof is important cuz Matlab

    env_up_outside_lesion = env_up[mask_out]
    mean_out = env_up_outside_lesion.mean()
    var_out = env_up_outside_lesion.var(ddof=1) # ddof is important cuz Matlab

    loggin_debug('{}: r6: Finished mean and var calculations'.format(target_dirname))
    CR = -20 * math_log10(mean_in / mean_out)
    CNR = 20 * math_log10(abs(mean_in - mean_out)/math_sqrt(var_in + var_out))
    SNR = mean_out / math_sqrt(var_out)

    loggin_debug('{}: r6: Finished speckle stats calculations'.format(target_dirname))
    # Save image statistics to file
    speckle_stats = [CR, CNR, SNR, mean_in, mean_out, var_in, var_out]
    speckle_stats_path = os_path_join(target_dirname, SPECKLE_STATS_FNAME)

    with open(speckle_stats_path, 'w') as f:
        f.write("\n".join([str(item) for item in speckle_stats]))

    loggin_debug('{}: r6: Finished saving .txt'.format(target_dirname))
    # Also save image statistics json as a redundant (but more readable) method
    speckle_stats_dict = {
        'CR': CR,
        'CNR': CNR,
        'SNR': SNR,
        'mean_inside_lesion': mean_in,
        'variance_inside_lesion': var_in,
        'mean_outside_lesion': mean_out,
        'variance_outside_lesion': var_out,
    }

    speckle_stats_dict_path = os_path_join(target_dirname, SPECKLE_STATS_DICT_FNAME)

    with open(speckle_stats_dict_path, 'w') as f:
        json_dump(speckle_stats_dict, f, indent=4)
    loggin_debug('{}: r6: Finished saving .json'.format(target_dirname))

    logging_info('{}: r6 Done'.format(target_dirname))


# def get_circular_mask(xx, yy, circle_center, circle_radius):
#     assert xx.shape == yy.shape
#     circle_coords_x, circle_coords_y = circle_center
#     return np_sqrt((xx-circle_coords_x) ** 2 + (yy-circle_coords_y) ** 2) <= circle_radius


# def get_rectangle_mask(xx, yy, xmin, xmax, ymin, ymax):
#     return (xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax)


def get_masks(target_dirname):
    masks = np_load(os_path_join(target_dirname, MASKS_FNAME))
    return masks['mask_in'], masks['mask_out']
