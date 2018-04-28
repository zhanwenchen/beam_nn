# Approximating Low-Pass Filters with Neural Networks

## Run instructions

The preferred way to run the experiment is through `Play.ipynb`. The following
are instructions for using a remote machine with CUDA, assuming you have
ssh access to the said machine.


### 1. On the remote machine

On a server, (probably not an HPC cluster and definitely not on Vanderbilt's
ACCRE), open `Play.ipynb` with Jupyter Notebook without a browser:

```sh
jupyter notebook Play.ipynb --no-browser
```

Don't leave yet - Jupyter will find an open port and give you a token string.
Remember these for the next step. Let's say the port is `8893` and the token is
`blahBlahBlah`.

### 2. On your local machine

You want to


# CHANGES
1. (CUDA) Attempt to improve performance with `torch.backends.cudnn.benchmark = True`
1. (CUDA) Attempt to improve performance by adding `pin_memory=True` to DataLoader
1. (CUDA) Attempt to improve performance by adding `async=True` in fit.py
