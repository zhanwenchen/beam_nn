
# coding: utf-8

# In[1]:


from torch.utils.data import DataLoader
from torch import optim
from torch.backends import cudnn
import numpy as np
import time
import csv
import warnings

from lib.aperture_dataset import ApertureDataset
from lib.lenet import LeNet
from lib.fit import fit

log_fname = 'results_%s.csv' % time.strftime('%Y%m%d-%H%M%S')

k = 4
fname_train = 'training_data/training_data.h5'
fname_validate = 'training_data/validation_data.h5'
save_path = 'save_here'


num_samples = 10 ** 5
dat_train = ApertureDataset(fname_train, num_samples, k)

num_samples = 10 ** 4
dat_train2 = ApertureDataset(fname_train, num_samples, k)

num_samples = 10 ** 4
dat_validate = ApertureDataset(fname_validate, num_samples, k)


# In[ ]:


# TODO: Implement Dropout

using_cuda = True
if not using_cuda: warnings.warn("Not using CUDA")
if using_cuda: cudnn.benchmark = True # CHANGED

# optimization parameters
try_learning_rates = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001]
optimization_algo = "Momentum"
if optimization_algo == "Momentum": momentum = 0.9
# try_batch_sizes = [32, 64, 128]
batch_size = 32
# NOTE we are using early stopping where we stop training once validation cost stops decreasing.
train_loader = DataLoader(dat_train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True) # CHANGED
train_loader2 = DataLoader(dat_train2, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True) # CHANGED
validate_loader = DataLoader(dat_validate, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True) # CHANGED

# Model parameters
input_size = 65
output_size = 130

# conv1
try_conv1_kernel_sizess = list(range(2, 10))
try_conv1_num_kernels = list(range(2, 33))
try_conv1_strides = [1, 2]

# pool1
try_pool1_sizes = [2, 3]
# try_pool1_sizes = [2] # TODO: support pool_size = 3 in lenet.

# conv2
# TODO in loop: conv2_kernel_size should <= conv1_kernel_size
# TODO in loop: conv2_num_kernels should >= conv1_num_kernels
try_conv2_kernel_sizess = list(range(2, 10))
try_conv2_num_kernels = list(range(2, 33))
try_conv2_strides = [1, 2]

# pool2
try_pool2_sizes = [2, 3]

# fcs
try_fc_hidden_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
try_fc_num_hidden_layers = [1, 2, 3]
                    

# Random Hyperparameter Search
def choose_int(array): return int(np.random.choice(array)) # int() because PyTorch doesn't convert np.int64 to int.

num_trys = 100
run_counter = 0

print()
print("| #  | vali cost | diff% | learn_rate | batch_size | conv1_kernel_size | conv1_num_kernels | conv1_stride | pool1_kernels | conv2_kernel_size | conv2_kernels | conv2_strides | pool2_kernels | fcs_size | fcs_num_hidden_layers |")
print("| -- | --------- | ----- | ---------- | ---------- | ----------------- | ----------------- | ------------ | ------------- | ----------------- | ------------- | ------------- | ------------- | -------- | --------------------- |")

while run_counter < num_trys:
    # choose random hyperparameters: optimization
#     batch_size = choose_int(try_batch_sizes)
    learning_rate = np.random.choice(try_learning_rates)
    
    # choose random hyperparameters: model
    conv1_kernel_size = choose_int(try_conv1_kernel_sizess)
    conv1_num_kernels = choose_int(try_conv1_num_kernels)
    conv1_stride = choose_int(try_conv1_strides)
    
    # enforce relative shape and divisibility
    conv1_output_size = (conv1_num_kernels, (input_size - conv1_kernel_size) / conv1_stride + 1)
    while isinstance(conv1_output_size[1], float) and not conv1_output_size[1].is_integer():
        # conv1_output_size must be an integer
        # random search is not the most efficient approach but I'm too lazy to filter right now.
        conv1_stride = choose_int(try_conv1_strides)
        conv1_output_size = (conv1_num_kernels, (input_size - conv1_kernel_size) / conv1_stride + 1)
#         print('loop: conv1_output_size =', conv1_output_size)
        
    
    pool1_kernel_size = choose_int(try_pool1_sizes)
    pool1_stride = 2
#     print("loop: (conv1_output_size[1] - pool1_kernel_size) =", (conv1_output_size[1] - pool1_kernel_size))
    pool1_output_size = (conv1_num_kernels, (conv1_output_size[1] - pool1_kernel_size) / pool1_stride + 1)
    while isinstance(pool1_output_size[1], float) and not pool1_output_size[1].is_integer():
        # conv1_output_size must be an integer
        pool1_kernel_size = choose_int(try_pool1_sizes)
        pool1_output_size = (conv1_num_kernels, (conv1_output_size[1] - pool1_kernel_size) / pool1_stride + 1)
#         print("loop: (conv1_output_size[1] - pool1_kernel_size) =", (conv1_output_size[1] - pool1_kernel_size))
#         print('loop: pool1_output_size =', pool1_output_size)
    
    conv2_kernel_size = choose_int(try_conv2_kernel_sizess)
    conv2_num_kernels = choose_int(try_conv2_num_kernels)
    conv2_stride = choose_int(try_conv2_strides)
    
    conv2_output_size = (conv2_num_kernels, (pool1_output_size[1] - conv2_kernel_size) / conv2_stride + 1)
#     print("loop: conv2_output_size = (%s - %s ) / %s + 1 = %s" % (pool1_output_size[1], conv2_kernel_size, conv2_stride, conv2_output_size[1]))
    while isinstance(conv2_output_size[1], float) and not conv2_output_size[1].is_integer():
        # conv2_output_size must be an integer
        conv2_stride = choose_int(try_conv2_strides)
        conv2_output_size = (conv2_num_kernels, (pool1_output_size[1] - conv2_kernel_size) / conv2_stride + 1)
#         print("loop: conv2_output_size = (%s - %s ) / %s + 1 = %s" % (pool1_output_size[1], conv2_kernel_size, conv2_stride, conv2_output_size[1]))
#         print('loop: conv2_output_size =', conv2_output_size)
    
#     print("loop: conv2: conv2_kernel_size = %s, conv2_num_kernels = %s, conv2_stride = %s" % (conv2_kernel_size, conv2_num_kernels, conv2_stride))

    pool2_kernel_size = choose_int(try_pool2_sizes)
    pool2_stride = 2
    pool2_output_size = (conv2_num_kernels, (conv2_output_size[1] - pool2_kernel_size) / pool2_stride + 1)
    
    while isinstance(pool2_output_size[1], float) and not pool2_output_size[1].is_integer():
        # conv1_output_size must be an integer
        pool2_kernel_size = choose_int(try_pool2_sizes)
        pool2_output_size = (conv2_num_kernels, (conv2_output_size[1] - pool2_kernel_size) / pool2_stride + 1)
#         print('loop: pool2_output_size =', pool2_output_size)
    
    fcs_hidden_size = choose_int(try_fc_hidden_sizes)
    fcs_num_hidden_layers = choose_int(try_fc_num_hidden_layers)
    
    # load data because of difference batch_sizes
#     train_loader = DataLoader(dat_train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=False) # CHANGED
#     train_loader2 = DataLoader(dat_train2, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False) # CHANGED
#     validate_loader = DataLoader(dat_validate, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False) # CHANGED
    
    # build the model and train
    model = LeNet(input_size, output_size, fcs_hidden_size, fcs_num_hidden_layers,
                  batch_size,
                  pool1_kernel_size,
                  conv1_kernel_size, conv1_num_kernels, conv1_stride,
                  pool2_kernel_size,
                  conv2_kernel_size, conv2_num_kernels, conv2_stride)
    
    if using_cuda: model.cuda()
    optimizer = optim.SGD(model.parameters(), learning_rate, momentum=momentum)
    
    cost, diff = fit(model, train_loader, train_loader2, validate_loader, optimizer, save_path, cuda=using_cuda)
    
    training_time = time.time() - time_before_train

    # TODO also fcs_num_hidden_layers
    print("| %02d | %.7f | %0.3f | %0.7f  | %03d        | %02d                | %02d                 | %d            | %d             | %d                | %d             | %d             | %d           | %04d     | %s |" %           (run_counter,cost, diff,learning_rate,batch_size,conv1_kernel_size,  conv1_num_kernels,      conv1_stride,   pool1_kernel_size,    conv2_kernel_size,  conv2_num_kernels,   conv2_stride, pool2_kernel_size,    fcs_hidden_size, fcs_num_hidden_layers))
    run_counter += 1

print("\nIt's all over.")

