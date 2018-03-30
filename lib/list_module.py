# A wrapper for a module list.
from torch import nn
class ListModule(object):
    #Should work with all kind of module
    # def __init__(self, module, prefix, *args):
    def __init__(self, module, prefix, *args): # CHANGED
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, key):
        if isinstance(key, slice) :
            #Get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int) :
            if key < 0 : #Handle negative indices
                key += len(self)
            if key < 0 or key >= len( self ) :
                raise IndexError("The index (%d) is out of range."%key)
            return getattr(self.module, self.prefix + str(key)) #Get the data from elsewhere
        else:
            raise TypeError("Invalid argument type.")
        # if i < 0 or i >= self.num_module:
        #     raise IndexError('Out of bound')
        # return getattr(self.module, self.prefix + str(i))
