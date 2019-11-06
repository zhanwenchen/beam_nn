# Versions

- 1.6.2 Narrowed learning rate to either 1e-04 or 1e-05.
- 1.6.1 Remove default LeakyReLU after last layer because regression should not have nonlinear output activation. Also added batch norm.

<!-- # Approximating Low-Pass Filters with Neural Networks -->

<!-- ## Project TODOs: -->


# Instructions

## Step 0. Set up
There are three directories excluded from Git, so you should set them up. The first is the `DNNs` folder. You should just `mkdir DNNs` under the project root. Then you need to download the `data` folder containing all training data folders, and download `scan_batteries` containing all the evaluation data folders. All three directories should be under project root.

## Step 1. Create Models
To use constraint satisfaction to find any number of models, you must install SWI-Prolog first in order to use `swipl`. After you have done this, call this command:

```sh
swipl create_fcns.pl 50 # Create 50 FCN models
```


## 2.  Training instructions

Requires package `h5py`.

Recommended concurrency is 4 because above 4 the GPU IO is bottlenecked.

```sh
python train.py "*" 4 # Train all created models, 4 at a time (concurrency)
python train.py "fcn_v1.6*" 3 # Train all created models with names starting with 'fcn_v1.6', 3 at a time (concurrency)
```

## 3. Evaluation instructions

Packages needed include `h5py` and a monkey-patched [`torchaudio`](https://github.com/zhanwenchen/audio/).

To evaluate trained models, do
```sh
python evaluate_models.py "*" # Evaluate all trained models that can be found
python evaluate_models.py "*" 1 # Evaluate 1 trained model, whichever one is found
python evaluate_models.py "fcn_v1.6.2_blah" # Evaluate model named 'fcn_v1.6.2_blah_trained', if it can be found
```

<!-- # Approach 1: Use create_50_models_and_train.sh
```sh
chmod +x create_50_models_and_train.sh
./create_50_models_and_train.sh
``` -->
<!--
# Approach 2: Create Any Number of Models (Using a Range Spec) and Train
Before training, under the project directory, create e.g. 50 models

```sh
pwd # should print ../beam_nn
Example: python lib/create_models.py 50 hyperparam_ranges.json
```

This creates 50 models under the beam_nn/DNNs folder.

Then train these models by running lib/main.py:

```sh
python lib/main.py DNNs/
```

This should start the training process!

## Evaluation Instructions

0. You must put 4 Matlab helper files under ~/matlab/helper
1. Make sure you have the correct beam_nn/scan_batteries folder structure like:

scan_batteries
- target_anachoic_cyst
- target_phantom_cyst
- target_in_vivo

2. Make sure that the process_scripts/r3_dnn_apply.py under each folder have 'lib' instead of 'src' and the correct LeNet init function.

3. Then, ensure permissions for the entire project. Otherwise, `python r3_dnn_apply.py` (in the process scripts under lib/) will throw a permission error, making remaining operations garbage. To do this, do

```sh
chown -R . && chmod -R +x *
```

4. get an identifier (for example, 20180705140109), and then run the evaluation file
```sh
python lib/evaluate_models.py 20180705140109
```

# CHANGES
1. (CUDA) Attempt to improve performance with `torch.backends.cudnn.benchmark = True`
1. (CUDA) Attempt to improve performance by adding `pin_memory=True` to DataLoader
1. (CUDA) Attempt to improve performance by adding `async=True` in fit.py -->
