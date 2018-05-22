
class Logger:

    def __init__(self):
        self.entries = {}

    def __repr__(self):
        return self.__str__()

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry

    def __str__(self):
        return str(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]

    def append(self, path):

        epoch = len(self.entries)
        line = [epoch]
        line.append( self.entries[epoch]['loss_train'] )
        line.append( self.entries[epoch]['loss_train_eval'] )
        line.append( self.entries[epoch]['loss_val'] )
        line = [str(item) for item in line]
        line = ','.join(line)
        line += '\n'

        f = open(path, 'a')
        f.write(line)
        f.close()
