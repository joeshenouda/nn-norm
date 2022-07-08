from random import sample, shuffle
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset


# shallow neural relu network
class shallow_NN(nn.Module):
    def __init__(self, input_dim, num_neurons, num_output):
        super().__init__()
        self.algo = 'v0'
        self.linear1 = nn.Linear(input_dim, num_neurons)
        self.linear2 = nn.Linear(num_neurons, num_output)

        self.grouped_layers = [[self.linear1, self.linear2]]
        self.other_layers = []
        self.num_neurons = [num_neurons]

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.linear1(x))
        out = self.linear2(x)

        return out

class myRNNLDataset(Dataset):
    def __init__(self, tensor_data, tensor_labels, transform=None):
        self.data=tensor_data
        tot_samps = self.data.shape[0]
        self.labels = tensor_labels
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        return sample, label

class RNNL:
    def __init__(self, args):
        super(RNNL, self).__init__()

        # use args:
        use_cuda = args.cuda
        num_workers = args.num_workers
        path = args.data_path
        batch_size = args.batch_size
        
        # basic info
        self.input_dim = int(np.sqrt(args.rnnl_dim))
        self.input_channel = 1
        self.num_classes = args.rnnl_out_dim
        self.num_samples = args.rnnl_samples

        self.rnnl_neurons = args.rnnl_neurons
        # Generate training set by constructing a neural network and passing random gaussian x;s into it

        # Generate X inputs
        #torch.seed(42)
        X = torch.randn(self.num_samples, args.rnnl_dim)

        # Construct shallow nn
        net = shallow_NN(int(self.input_dim**2), self.rnnl_neurons, self.num_classes)
        net.eval()
        Y = net(X)

        train_set = myRNNLDataset(X, Y.detach())
        PATH_train_set = os.path.join(args.dest_dir, 'training_set.pt')
        torch.save(train_set, PATH_train_set)

        kwargs = {"num_workers": num_workers, "pin_memory": True} if use_cuda else {}

        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)

