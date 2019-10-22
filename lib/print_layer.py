from torch.nn import Module


class PrintLayer(Module):
    def __init__(self, prev_layer):
        super(PrintLayer, self).__init__()
        self.prev_layer = prev_layer

    def forward(self, x):
        print('After prev_layer = {}, x.size() = {}'.format(self.prev_layer, x.size()))
        return x
