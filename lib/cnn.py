# This is option 1 of CNN - use one channel (130 * 1)
# TODO: Option 2: use two channels - real vs imaginery components each (65 * 2)

import torch
import torch.nn.functional as F
import torch.nn as nn

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(CNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.c1 = nn.Conv1d(input_size, hidden_size, 1)
        # self.p1 = nn.AvgPool1d(2)
        self.c2 = nn.Conv1d(hidden_size, hidden_size, 1)
        # self.p2 = nn.AvgPool1d(2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, input_size = x.size()
        x = x.unsqueeze(2)
        x = self.c1(x)
        x = self.c2(x)
        x = F.tanh(x)
        x = x.view(batch_size, self.hidden_size)
        x = self.fc(x)
        x = F.tanh(x)
        return x
