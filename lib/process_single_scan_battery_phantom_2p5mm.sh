#!/bin/bash

# change into directory
echo $1
cd $1

if [ ! -d 'scan_batteries' ]; then
    mkdir scan_batteries
fi

cd scan_batteries

echo "Copying scan battery"
cp -r ../../../scan_batteries/target_phantom_anechoic_cyst_2p5mm ./

cd target_phantom_anechoic_cyst_2p5mm
echo "Current directory:"
echo $(pwd)

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

        echo "Processing " $dir
        cd $dir

        # process with networks
        echo "r3_dnn_apply.py"
        python ../process_scripts/r3_dnn_apply.py

        # take istft of chandat, create dnn image data, display dnn image
        echo "r4_dnn_istft"
        echo "r5_dnn_image"
        echo "r6_dnn_image_display"
        matlab -nosoftwareopengl -nodesktop -nosplash -r "addpath('../process_scripts'); r4_dnn_istft; r5_dnn_image; r6_dnn_image_display; quit;"

        # delete extra files
        echo "Deleting old_stft.mat"
        rm old_stft.mat
        echo "Deleting new_stft.mat"
        rm new_stft.mat
        echo "Deleting chandat.mat"
        rm chandat.mat
        echo "Deleteing chandat_dnn.mat"
        rm chandat_dnn.mat
        echo "Deleteing dnn_image.mat"
        rm dnn_image.mat
        echo "Removing ROI files"
        rm box_*
        rm circle_*

        cd ..

    done
done

#echo "Deleting process_scripts"
rm -r process_scripts
