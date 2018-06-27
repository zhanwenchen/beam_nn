# Approximating Low-Pass Filters with Neural Networks

## Run instructions

The preferred way to run the experiment is through `lib/main.py`. The following
are instructions for using a remote machine with CUDA, assuming you have
ssh access to the said machine.

### 2. On your local machine

For example, you can specify a single DNN for a specific k value like

```sh
python lib/main.py DNNs/1530127690_1/k_3/model_params.txt
```


# CHANGES
1. (CUDA) Attempt to improve performance with `torch.backends.cudnn.benchmark = True`
1. (CUDA) Attempt to improve performance by adding `pin_memory=True` to DataLoader
1. (CUDA) Attempt to improve performance by adding `async=True` in fit.py
