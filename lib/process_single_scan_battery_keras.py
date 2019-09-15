from os import remove as os_remove
from os.path import join as os_path_join, \
                    basename as os_path_basename, \
                    isdir as os_path_isdir, \
                    isfile as os_path_isfile
from multiprocessing import Pool
from glob import glob as glob_glob
import shutil
from logging import info as logging_info

from scipy.io import loadmat

from lib.r3_dnn_apply_keras import r3_dnn_apply_keras
from lib.r4_dnn_istft import r4_dnn_istft
from lib.r5_dnn_image import r5_dnn_image
from lib.r6_dnn_image_display import r6_dnn_image_display

SCRIPT_FNAME = os_path_basename(__file__) # for error messages. File name can change.

CHANDAT_FNAME = 'chandat.mat'
SCAN_BATTERIES_DIRNAME = 'scan_batteries'
TARGET_PREFIX = 'target_'
PROCESS_SCRIPTS_DIRNAME = 'process_scripts'
TARGET_FILES_TO_REMOVE = ['old_stft.mat', 'new_stft.mat', 'chandat.mat', 'chandat_dnn.mat', 'dnn_image.mat', 'box_left_max.txt', 'box_left_min.txt', 'box_right_max.txt', 'box_right_min.txt', 'circle_out_radius.txt', 'circle_out_xc.txt', 'circle_out_zc.txt', 'region_in.txt', 'circle_radius.txt', 'circle_xc.txt', 'circle_zc.txt']
SCAN_BATTERY_FOLDERS_TO_REMOVE = ['process_scripts', 'creation_scripts', 'phantoms']
SCAN_BATTERY_FILES_TO_REMOVE = ['delete_files.sh', 'folders_in_battery.txt', 'model_dirs.txt']


def process_single_scan_battery_keras(model_folder, source_scan_battery_dirname):
    # Make sure model_folder and source_scan_battery_dirname exist.
    if not os_path_isdir(model_folder):
        raise OSError('{}: model folder {} does not exist'.format(SCRIPT_FNAME, model_folder))
    if not os_path_isdir(source_scan_battery_dirname):
        raise OSError('{}: source scan battery folder {} does not exist'.format(SCRIPT_FNAME, source_scan_battery_dirname))

    # model/scan_batteries folders.
    model_scan_batteries_dirname = os_path_join(model_folder, SCAN_BATTERIES_DIRNAME)
    model_scan_battery_dirname = os_path_join(model_scan_batteries_dirname, os_path_basename(source_scan_battery_dirname))

    # Copy source scan_batteries folder into model scan_batteries folder
    # TODO: Could also just copy the entire scan_batteries folder (all 3 types) into model_folder
    # logging_info('{}: copying {} to {}'.format(SCRIPT_FNAME, source_scan_battery_dirname, model_scan_battery_dirname))
    # copy_anything(source_scan_battery_dirname, model_scan_battery_dirname)
    # model_scan_battery_process_scripts_dirname = os_path_abspath(os_path_join(model_scan_battery_dirname, PROCESS_SCRIPTS_DIRNAME))

    # Grab all targets with glob
    mode_scan_battery_target_prefix = os_path_join(model_scan_battery_dirname, TARGET_PREFIX + '*')
    target_dirnames = glob_glob(mode_scan_battery_target_prefix)
    if not target_dirnames:
        raise ValueError('{}: no targets found with prefix {}'.format(SCRIPT_FNAME, mode_scan_battery_target_prefix))

    with Pool() as pool:
        list(pool.imap_unordered(process_single_target, target_dirnames))

    # for target_dirname in target_dirnames:
    #     # print('{}: processing target directory {}'.format(SCRIPT_FNAME, target_dirname))
    #     process_single_target(target_dirname)

    # Remove scan battery-level folders
    for folder in SCAN_BATTERY_FOLDERS_TO_REMOVE:
        folder_path = os_path_join(model_scan_battery_dirname, folder)
        if os_path_isdir(folder_path):
            # print('{}: Trying to remove {}'.format(SCRIPT_FNAME, folder_path))
            try:
                shutil.rmtree(folder_path)
            except:
                raise OSError('Error: unable to remove file {}'.format(folder_path))

    # Remove scan battery-level files
    for file in SCAN_BATTERY_FILES_TO_REMOVE:
        file_path = os_path_join(model_scan_battery_dirname, file)
        if os_path_isfile(file_path):
            # print('{}: Trying to remove {}'.format(SCRIPT_FNAME, file_path))
            try:
                os_remove(file_path)
            except:
                raise OSError('Error: unable to remove file {}'.format(file_path))


def process_single_target(target_dirname):
    chandat_obj = loadmat(os_path_join(target_dirname, CHANDAT_FNAME))
    new_stft_object = r3_dnn_apply_keras(target_dirname, saving_to_disk=False)
    chandat_dnn_object = r4_dnn_istft(target_dirname, chandat_obj=chandat_obj, new_stft_object=new_stft_object, is_saving_chandat_dnn=False)
    chandat_image_obj = r5_dnn_image(target_dirname, chandat_obj=chandat_obj, chandat_dnn_obj=chandat_dnn_object, is_saving_chandat_image=False)
    r6_dnn_image_display(target_dirname, dnn_image_obj=chandat_image_obj, show_fig=False)

    # Remove target-level files and folders
    for file in TARGET_FILES_TO_REMOVE:
        file_path = os_path_join(target_dirname, file)
        if os_path_isfile(file_path):
            # print('{}: Trying to remove {}'.format(SCRIPT_FNAME, file_path))
            try:
                os_remove(file_path)
            except:
                raise OSError('Error: unable to remove file {}'.format(file_path))
