from os.path import join as os_path_join
from math import log10 as math_log10, sqrt as math_sqrt
from json import dump as json_dump

from scipy.io import loadmat
from numpy import meshgrid as np_meshgrid
from numpy import sqrt as np_sqrt
from numpy import squeeze as np_squeeze
from matplotlib import pyplot as plt


DNN_IMAGE_FNAME = 'dnn_image.mat'
FONT_SIZE = 20
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


def r6_dnn_image_display(target_dirname, show_fig=False):
    dnn_image_path = os_path_join(target_dirname, DNN_IMAGE_FNAME)
    dnn_image_obj = loadmat(dnn_image_path)

    beam_position_x_up = dnn_image_obj['beam_position_x_up']
    depth = dnn_image_obj['depth']
    envUp_dB = dnn_image_obj['envUp_dB']
    env_up = dnn_image_obj['envUp']

    x = beam_position_x_up = np_squeeze(beam_position_x_up)
    y = depth = np_squeeze(depth)

    fig = plt.figure(figsize=(12,16))
    plt.imshow(envUp_dB, vmin=-60, vmax=0, cmap='gray', aspect='auto', extent=[beam_position_x_up[0]*1000,beam_position_x_up[-1]*1000,depth[-1]*1000, depth[0]*1000])
    plt.colorbar()
    plt.xlabel('lateral (mm)', fontsize=FONT_SIZE)
    plt.ylabel('axial (mm)', fontsize=FONT_SIZE)

    if show_fig is True:
        plt.show()

    # Save image to file
    dnn_image_path = os_path_join(target_dirname, DNN_IMAGE_SAVE_FNAME)
    fig.savefig(dnn_image_path)

    circle_radius = load_single_value(target_dirname, CIRCLE_RADIUS_FNAME)
    circle_coords_x = load_single_value(target_dirname, CIRCLE_COORDS_X_FNAME)
    circle_coords_y = load_single_value(target_dirname, CIRCLE_COORDS_Y_FNAME)

    xx, yy = np_meshgrid(x, y)
    mask_in = get_circular_mask(xx, yy, (circle_coords_x, circle_coords_y), circle_radius)

    # create rectangular region outside lesion
    box_xmin_right = load_single_value(target_dirname, BOX_XMIN_RIGHT_FNAME)
    box_xmax_right = load_single_value(target_dirname, BOX_XMAX_RIGHT_FNAME)

    box_xmin_left = load_single_value(target_dirname, BOX_XMIN_LEFT_FNAME)
    box_xmax_left = load_single_value(target_dirname, BOX_XMAX_LEFT_FNAME)

    # Box shares y position and height with circle (diameter)
    ymin = circle_coords_y - circle_radius
    ymax = circle_coords_y + circle_radius
    mask_out_left = (xx >= box_xmin_left) * (xx <= box_xmax_left) * (yy >= ymin) * (yy <= ymax)
    mask_out_right = get_rectangle_mask(xx, yy, box_xmin_right, box_xmax_right, ymin, ymax)
    mask_out = mask_out_left | mask_out_right

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
    env_up_inside_lesion = env_up[mask_in]
    mean_in = env_up_inside_lesion.mean()
    var_in = env_up_inside_lesion.var(ddof=1) # ddof is important cuz Matlab

    env_up_outside_lesion = env_up[mask_out]
    mean_out = env_up_outside_lesion.mean()
    var_out = env_up_outside_lesion.var(ddof=1) # ddof is important cuz Matlab

    CR = -20 * math_log10(mean_in / mean_out)
    CNR = 20 * math_log10(abs(mean_in - mean_out)/math_sqrt(var_in + var_out))
    SNR = mean_out / math_sqrt(var_out)

    # Save image statistics to file
    speckle_stats = [CR, CNR, SNR, mean_in, mean_out, var_in, var_out]
    speckle_stats_path = os_path_join(target_dirname, SPECKLE_STATS_FNAME)

    with open(speckle_stats_path, 'w') as f:
        f.write("\n".join([str(item) for item in speckle_stats]))

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


# TODO: Move utils to another file
def load_single_value(target_dirname, fname):
    path = os_path_join(target_dirname, fname)

    with open(path, 'r') as f:
        value = float(f.read())

    return value


def get_circular_mask(xx, yy, circle_center, circle_radius):
    assert xx.shape == yy.shape
    circle_coords_x, circle_coords_y = circle_center
    return np_sqrt((xx-circle_coords_x) ** 2 + (yy-circle_coords_y) ** 2) <= circle_radius


def get_rectangle_mask(xx, yy, xmin, xmax, ymin, ymax):
    return (xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax)
