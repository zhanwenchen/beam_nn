
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

num_samples = 10 ** 5
dat_train = ApertureDataset(fname_train, num_samples, 4)
train_loader = DataLoader(dat_train, batch_size=1000, shuffle=True, num_workers=1)

num_samples = 10 ** 4
dat_train2 = ApertureDataset(fname_train, num_samples, 4)
train_loader2 = DataLoader(dat_train2, batch_size=1000, shuffle=False, num_workers=1)

num_samples = 10 ** 4
dat_validate = ApertureDataset(fname_validate, num_samples, 4)
validate_loader = DataLoader(dat_validate, batch_size=1000, shuffle=False, num_workers=1)


# In[2]:


from torch import optim

# from lib.fully_connected_net import FullyConnectedNet
from lib.fully_connected_net import FullyConnectedNet
from lib.fit import fit

lr = 0.5
momentum = 0
nn_fc = FullyConnectedNet(**model_params_dict)
optimizer = optim.SGD(nn_fc.parameters(), lr, momentum)

fit(nn_fc, train_loader, train_loader2, validate_loader, optimizer, save_path, cuda=False)


# In[ ]:
