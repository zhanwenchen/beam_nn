# Approximating Low-Pass Filters with Neural Networks

## Project TODOs:

-[] Try Dropout1d instead of current Dropout2d
-[] Try Adam and SGD

## Training instructions


# Approach 1: Use create_50_models_and_train.sh
```sh
chmod +x create_50_models_and_train.sh
./create_50_models_and_train.sh
```

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
1. (CUDA) Attempt to improve performance by adding `async=True` in fit.py
