#!/bin/bash
set -e

lib_dir=$(pwd)
# change into directory
echo process_single_scan_battery_phantom_2p5mm.sh: $1
cd $1

if [ ! -d 'scan_batteries' ]; then
    mkdir scan_batteries
fi

cd scan_batteries

echo "process_single_scan_battery_phantom_2p5mm.sh: Copying scan battery"
cp -r ../../../scan_batteries/target_phantom_anechoic_cyst_2p5mm ./

cd target_phantom_anechoic_cyst_2p5mm
echo "process_single_scan_battery_phantom_2p5mm.sh: Current directory is " $(pwd)

# process stft data with neural networks
#echo "r3_dnn_apply_battery.py"
#python process_scripts/r3_dnn_apply_battery.py --cuda

# form images from dnn data
for i in {1,2,3,4,5}
do
    for m in $(seq 1 1 1)
    do
        # directory to analyze
        dir='target_'
        dir+=$i

        echo "process_single_scan_battery_phantom_2p5mm.sh: Processing " $dir
        cd $dir

        # process with networks
        echo "process_single_scan_battery_phantom_2p5mm.sh: running r3_dnn_apply.py"
        # python ../process_scripts/r3_dnn_apply.py
        python "$lib_dir/r3_dnn_apply.py" -c

        # take istft of chandat, create dnn image data, display dnn image
        echo "process_single_scan_battery_phantom_2p5mm.sh: Matlab running r4_dnn_istft.m, r5_dnn_image.m, and r6_dnn_image_display.m"
        matlab -nosoftwareopengl -nodesktop -nosplash -r "try, addpath('../process_scripts'), r4_dnn_istft, r5_dnn_image, r6_dnn_image_display, catch, exit(1), end, exit(0);"
        # matlab -nodesktop -nosplash -r "try, addpath('../process_scripts'), r4_dnn_istft, r5_dnn_image, r6_dnn_image_display, catch, exit(1), end, exit(0);"

        # delete extra files
        echo "process_single_scan_battery_phantom_2p5mm.sh: Deleting old_stft.mat"
        rm old_stft.mat
        echo "process_single_scan_battery_phantom_2p5mm.sh: Deleting new_stft.mat"
        rm new_stft.mat
        echo "process_single_scan_battery_phantom_2p5mm.sh: Deleting chandat.mat"
        rm chandat.mat
        echo "process_single_scan_battery_phantom_2p5mm.sh: Deleteing chandat_dnn.mat"
        rm chandat_dnn.mat
        echo "process_single_scan_battery_phantom_2p5mm.sh: Deleteing dnn_image.mat"
        rm dnn_image.mat
        echo "process_single_scan_battery_phantom_2p5mm.sh: Removing ROI files"
        rm box_*
        rm circle_*

        cd ..

    done
done

echo "process_single_scan_battery_phantom_2p5mm.sh: Deleting process_scripts"
rm -r process_scripts
