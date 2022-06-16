import torch
from torch import nn
import torch.nn.functional as F


# shallow neural relu network with soft-max
class shallow_NN(nn.Module):
    def __init__(self, input_dim, input_channel, num_hidden, num_classes):
        super().__init__()
        self.algo = 'v0'
        self.linear1 = nn.Linear(input_dim * input_dim * input_channel, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_classes)

        self.grouped_layers = [[self.linear1, self.linear2]]
        self.other_layers = []
        self.num_neurons = [num_hidden]

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.linear1(x))
        out = self.linear2(x)

        return out

