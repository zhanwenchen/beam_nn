# Approximating Low-Pass Filters with Neural Networks

## Run instructions

Before training, under the project directory, create e.g. 50 models

```sh
pwd # should print ../beam_nn
python lib/create_hyperparam_search_old.py 50
```

This creates 50 models under the beam_nn/DNNs folder.

Then train these models by running lib/main.py:

```sh
python lib/main.py DNNs/
```

This should start the training process!


# CHANGES
1. (CUDA) Attempt to improve performance with `torch.backends.cudnn.benchmark = True`
1. (CUDA) Attempt to improve performance by adding `pin_memory=True` to DataLoader
1. (CUDA) Attempt to improve performance by adding `async=True` in fit.py
