
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

from torch.utils.data import DataLoader

from lib.aperture_dataset import ApertureDataset

fname_train = 'training_data/training_data.h5'
fname_validate = 'training_data/validation_data.h5'

save_path = 'save_here'

model_params_dict = {
    'input_dim': 130, # TODO: move?
    'output_dim': 130,
    'layer_width': 260,
    'num_hidden_layers': 5,
}

input_dim = model_params_dict['input_dim']
output_dim = model_params_dict['output_dim']
layer_width = model_params_dict['layer_width']
num_hidden_layers = model_params_dict['num_hidden_layers']
batch_size = 32

num_samples = 10 ** 5
dat_train = ApertureDataset(fname_train, num_samples, 4)
train_loader = DataLoader(dat_train, batch_size=batch_size, shuffle=True, num_workers=1)

num_samples = 10 ** 4
dat_train2 = ApertureDataset(fname_train, num_samples, 4)
train_loader2 = DataLoader(dat_train2, batch_size=batch_size, shuffle=False, num_workers=1)

num_samples = 10 ** 4
dat_validate = ApertureDataset(fname_validate, num_samples, 4)
validate_loader = DataLoader(dat_validate, batch_size=batch_size, shuffle=False, num_workers=1)


# In[2]:


from torch import optim

# from lib.fully_connected_net import FullyConnectedNet
from lib.fully_connected_net import FullyConnectedNet
from lib.cnn import CNN
from lib.lenet import LeNet
from lib.fit import fit

using_cuda = False

lr = 0.5
momentum = 0

# fc = FullyConnectedNet(**model_params_dict)
# optimizer_fc = optim.SGD(fc.parameters(), lr, momentum)
# cnn = CNN(input_dim, output_dim, layer_width)
# optimizer_cnn = optim.SGD(cnn.parameters(), lr, momentum)
lenet = LeNet(input_dim, output_dim, layer_width, batch_size)

model = lenet
optimizer = optim.SGD(model.parameters(), lr, momentum)

if using_cuda:
    model.cuda()

print(model)
# fit(nn_fc, train_loader, train_loader2, validate_loader, optimizer_fc, save_path, cuda=False)
fit(model, train_loader, train_loader2, validate_loader, optimizer, save_path, cuda=using_cuda)
# In[ ]:
