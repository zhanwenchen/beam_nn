#!/bin/bash
set -e

# change into directory
echo $1
cd $1

if [ ! -d 'scan_batteries' ]; then
    mkdir scan_batteries
fi

cd scan_batteries

echo "process_single_scan_battery_anechoic_cyst.sh: Copying scan battery"
cp -r ../../../scan_batteries/target_anechoic_cyst_5mm ./

cd target_anechoic_cyst_5mm
echo "process_single_scan_battery_anechoic_cyst.sh: Current directory is " $(pwd)

# process stft data with neural networks
#echo "r3_dnn_apply_battery.py"
#python process_scripts/r3_dnn_apply_battery.py --cuda

# form images from dnn data
for i in $(seq 1 1 5)
do
    for m in $(seq 1 1 1)
    do

        # directory to analyze
        dir='target_'
        dir+=$i
        dir+='_SNR_10dB'

        echo "process_single_scan_battery_anechoic_cyst.sh: Processing " $dir
        cd $dir

        # dnn processing
        echo "process_single_scan_battery_anechoic_cyst.sh: DNN processing"
        python ../process_scripts/r3_dnn_apply.py

        # take istft of chandat, create dnn image data, display dnn image
        echo "process_single_scan_battery_anechoic_cyst.sh: Matlab run r4_dnn_istft.m, r5_dnn_image.m, and r6_dnn_image_display.m"
        # matlab -nosoftwareopengl -nodesktop -nosplash -r "addpath('../process_scripts'); r4_dnn_istft; r5_dnn_image; r6_dnn_image_display; quit;"
        matlab -nosoftwareopengl -nodesktop -nosplash -r "try, addpath('../process_scripts'), r4_dnn_istft, r5_dnn_image, r6_dnn_image_display, catch, exit(1), end, exit(0);"

        # delete extra files
        echo "process_single_scan_battery_anechoic_cyst.sh: Deleting old_stft.mat"
        rm old_stft.mat
        echo "process_single_scan_battery_anechoic_cyst.sh: Deleting new_stft.mat"
        rm new_stft.mat
        echo "process_single_scan_battery_anechoic_cyst.sh: Deleting chandat.mat"
        rm chandat.mat
        echo "process_single_scan_battery_anechoic_cyst.sh: Deleteing chandat_dnn.mat"
        rm chandat_dnn.mat
        echo "process_single_scan_battery_anechoic_cyst.sh: Deleteing dnn_image.mat"
        rm dnn_image.mat

        cd ..

    done
done

echo "process_single_scan_battery_anechoic_cyst.sh: Deleting process_scripts, creation_scripts, and phantom folders"
rm -r process_scripts creation_scripts phantoms
