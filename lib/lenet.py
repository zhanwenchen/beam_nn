# This is option 1 of CNN - use one channel (130 * 1)
# TODO: Option 2: use two channels - real vs imaginery components each (65 * 2)

import torch
import torch.nn.functional as F
import torch.nn as nn

# CNN Model (2 conv layer)
class LeNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, batch_size):
        super(LeNet, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.conv1 = nn.Conv1d(1, 1, 2) # NOTE: THIS IS CORRECT!!!! CONV doesn't depend on num_features!
        # self.pool1 = nn.AvgPool1d(2)

        self.conv2 = nn.Conv1d(1, 1, 2)
        # self.pool2 = nn.AvgPool1d(2)

        self.fc1 = nn.Linear(128, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        batch_size, input_size = x.size()
        x = x.unsqueeze(1) # right now it's (32, 130, 1). Should be (130, 32, 1) or (1, 32, 130). NOTE it's not (1, 32, 130) or (32, 1, 130).  It has to be (1, 130, 32)
        x = self.conv1(x)
        x = F.relu(x)
        # x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        # x = self.pool2(x)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
