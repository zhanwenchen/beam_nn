from torch import prod, tensor
from torch.nn import Module


class Flatten(Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = prod(tensor(x.shape[1:])).item()
        return x.view(-1, shape)
