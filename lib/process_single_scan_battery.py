import os
import glob
import io
import shutil
import sys

import matlab.engine

from lib.utils import copy_anything, clean_buffers
from lib.r3_dnn_apply import main as r3_dnn_apply


SCRIPT_FNAME = os.path.basename(__file__) # for error messages. File name can change.


SCAN_BATTERIES_DIRNAME = 'scan_batteries'
TARGET_PREFIX = 'target_'
PROCESS_SCRIPTS_DIRNAME = 'process_scripts'
TARGET_FILES_TO_REMOVE = ['old_stft.mat', 'new_stft.mat', 'chandat.mat', 'chandat_dnn.mat', 'dnn_image.mat', 'box_left_max.txt', 'box_left_min.txt', 'box_right_max.txt', 'box_right_min.txt', 'circle_out_radius.txt', 'circle_out_xc.txt', 'circle_out_zc.txt', 'region_in.txt', 'circle_radius.txt', 'circle_xc.txt', 'circle_zc.txt']
SCAN_BATTERY_FOLDERS_TO_REMOVE = ['process_scripts', 'creation_scripts', 'phantoms']
SCAN_BATTERY_FILES_TO_REMOVE = ['delete_files.sh', 'folders_in_battery.txt', 'model_dirs.txt']


def process_single_scan_battery(model_folder, source_scan_battery_dirname, matlab_session=None):
    # Make sure model_folder and source_scan_battery_dirname exist.
    if not os.path.isdir(model_folder):
        raise OSError('{}: model folder {} does not exist'.format(SCRIPT_FNAME, model_folder))
    if not os.path.isdir(source_scan_battery_dirname):
        raise OSError('{}: source scan battery folder {} does not exist'.format(SCRIPT_FNAME, source_scan_battery_dirname))

    # model/scan_batteries folders.
    model_scan_batteries_dirname = os.path.join(model_folder, SCAN_BATTERIES_DIRNAME)
    model_scan_battery_dirname = os.path.join(model_scan_batteries_dirname, os.path.basename(source_scan_battery_dirname))

    # Copy source scan_batteries folder into model scan_batteries folder
    # TODO: Could also just copy the entire scan_batteries folder (all 3 types) into model_folder
    # print('{}: copying {} to {}'.format(SCRIPT_FNAME, scan_batteries_dirname, model_scan_battery_dirname))
    copy_anything(source_scan_battery_dirname, model_scan_battery_dirname)
    model_scan_battery_process_scripts_dirname = os.path.abspath(os.path.join(model_scan_battery_dirname, PROCESS_SCRIPTS_DIRNAME))

    # Grab all targets with glob
    mode_scan_battery_target_prefix = os.path.join(model_scan_battery_dirname, TARGET_PREFIX + '*')
    target_dirnames = glob.glob(mode_scan_battery_target_prefix)
    if not target_dirnames:
        raise ValueError('{}: no targets found with prefix {}'.format(SCRIPT_FNAME, mode_scan_battery_target_prefix))

    # Initialize MATLAB engine if none shared and stdout/stderr buffers
    # print('{}: starting MATLAB engine'.format(SCRIPT_FNAME))
    eng = matlab_session or matlab.engine.start_matlab()
    out = io.StringIO()
    err = io.StringIO()

    # MATLAB add process_scripts
    clean_buffers(out, err)
    try:
        eng.addpath(model_scan_battery_process_scripts_dirname, nargout=0, stdout=out, stderr=err)
        sys.stdout.write(out.getvalue())
        sys.stderr.write(err.getvalue())
    except Exception as e:
        sys.stdout.write(out.getvalue())
        sys.stderr.write(err.getvalue())
        raise RuntimeError(str(e))

    for target_dirname in target_dirnames:
        # print('{}: processing target directory {}'.format(SCRIPT_FNAME, target_dirname))
        r3_dnn_apply(target_dirname)

        # MATLAB cd into target folder
        # print('{}: MATLAB cd into target folder'.format(SCRIPT_FNAME))
        clean_buffers(out, err)
        try:
            eng.cd(os.path.abspath(target_dirname), nargout=0, stdout=out, stderr=err)
            sys.stdout.write(out.getvalue())
            sys.stderr.write(err.getvalue())
        except Exception as e:
            sys.stdout.write(out.getvalue())
            sys.stderr.write(err.getvalue())
            raise RuntimeError(str(e))


        # 1. MATLAB DNN inverse short-time fourier transform
        # print('{}: MATLAB r4_dnn_istft'.format(SCRIPT_FNAME))
        clean_buffers(out, err)
        try:
            eng.r4_dnn_istft(nargout=0, stdout=out, stderr=err)
            sys.stdout.write(out.getvalue())
            sys.stderr.write(err.getvalue())
        except Exception as e:
            sys.stderr.write(err.getvalue())
            raise RuntimeError(str(e))

        # 2. MATLAB r5_dnn_image
        # print('{}: MATLAB r5_dnn_image'.format(SCRIPT_FNAME))
        clean_buffers(out, err)
        try:
            eng.r5_dnn_image(nargout=0, stdout=out, stderr=err)
            sys.stdout.write(out.getvalue())
            sys.stderr.write(err.getvalue())
        except Exception as e:
            sys.stdout.write(out.getvalue())
            sys.stderr.write(err.getvalue())
            raise RuntimeError(str(e))

        # 3. MATLAB r6_dnn_image_display
        # print('{}: MATLAB r6_dnn_image_display'.format(SCRIPT_FNAME))
        clean_buffers(out, err)
        try:
            eng.r6_dnn_image_display(nargout=0, stdout=out, stderr=err)
            sys.stdout.write(out.getvalue())
            sys.stderr.write(err.getvalue())
        except Exception as e:
            sys.stdout.write(out.getvalue())
            sys.stderr.write(err.getvalue())
            raise RuntimeError(str(e))

        # 4. MATLAB clear all variables
        # print('{}: MATLAB clear all'.format(SCRIPT_FNAME))
        try:
            eng.clear('all', nargout=0)
            sys.stdout.write(out.getvalue())
            sys.stderr.write(err.getvalue())
        except Exception as e:
            sys.stdout.write(out.getvalue())
            sys.stderr.write(err.getvalue())
            raise RuntimeError(str(e))

        # Remove target-level files and folders
        for file in TARGET_FILES_TO_REMOVE:
            file_path = os.path.join(target_dirname, file)
            if os.path.isfile(file_path):
                # print('{}: Trying to remove {}'.format(SCRIPT_FNAME, file_path))
                try:
                    os.remove(file_path)
                except Exception as e:
                    raise OSError('Error: unable to remove file {}'.format(file_path))

    # MATLAB remove process_scripts
    clean_buffers(out, err)
    try:
        eng.rmpath(model_scan_battery_process_scripts_dirname, nargout=0, stdout=out, stderr=err)
        sys.stdout.write(out.getvalue())
        sys.stderr.write(err.getvalue())
    except Exception as e:
        sys.stdout.write(out.getvalue())
        sys.stderr.write(err.getvalue())
        raise RuntimeError(str(e))

    # Remove scan battery-level folders
    for folder in SCAN_BATTERY_FOLDERS_TO_REMOVE:
        folder_path = os.path.join(model_scan_battery_dirname, folder)
        if os.path.isdir(folder_path):
            # print('{}: Trying to remove {}'.format(SCRIPT_FNAME, folder_path))
            try:
                shutil.rmtree(folder_path)
            except Exception as e:
                raise OSError('Error: unable to remove file {}'.format(folder_path))

    # Remove scan battery-level files
    for file in SCAN_BATTERY_FILES_TO_REMOVE:
        file_path = os.path.join(model_scan_battery_dirname, file)
        if os.path.isfile(file_path):
            # print('{}: Trying to remove {}'.format(SCRIPT_FNAME, file_path))
            try:
                os.remove(file_path)
            except Exception as e:
                raise OSError('Error: unable to remove file {}'.format(file_path))
