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
        entry = self.entries[epoch]
        line = [epoch, entry['loss_train'], entry['loss_train_eval'], entry['loss_val']]
        # line.append( entry['loss_train'] )
        # line.append( entry['loss_train_eval'] )
        # line.append( entry['loss_val'] )
        # line = [str(item) for item in line]
        line = ','.join([str(item) for item in line]) + '\n'
        # line += '\n'

        with open(path, 'a') as f:
            f.write(line)
