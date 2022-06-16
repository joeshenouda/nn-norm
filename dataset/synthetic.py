from torch.utils.data import Dataset
import numpy as np
import torch


def generate_sample(n):
    # generate the samples
    L = int(np.log2(n))
    xv = np.zeros(n)
    yv = np.zeros(n)
    for i in range(n):
        t = i / n
        xv[i] = t
        yv[i] = np.sin(2 * np.pi * t * 4) * (t > 0) * (t <= 1 / 2) + np.sin(2 * np.pi * (t - 1 / 2) * 16) * (t > 1 / 2) * (
                    t <= 3 / 4) + np.sin(2 * np.pi * (t - 3 / 4) * 4) * (t > 3 / 4) * (t <= 1);

    noise = np.random.normal(0, 1, n)
    yv = yv + .01 * noise

    x_train = torch.from_numpy(xv)[:, None].float()  # x_train shape [n, 1]
    y_train = torch.from_numpy(yv)[:, None].float()  # y_train shape [n, 1]
    return x_train, y_train

def generate_random_samples(n,d):
    x_train = torch.rand(50, d)
    y_train = torch.randn(50,2)*50

    return x_train, y_train

class synthetic_dataset(Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


class small_spatial:
    def __init__(self, args):
        super(small_spatial, self).__init__()

        # use args:
        use_cuda = args.cuda
        batch_size = args.batch_size
        n = args.num_samples

        # basic information
        self.input_dim = 1
        self.num_classes = 1

        # Data loading code
        kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
        x_train, y_train = generate_sample(n)
        self.x_train = x_train
        self.y_train = y_train
        dataset = synthetic_dataset(x_train, y_train)
        self.train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)

class random_data:
    def __init__(self, args):
        super(random_data, self).__init__()

        # use args:
        use_cuda = args.cuda
        batch_size = args.batch_size
        n = args.num_samples

        # basic information
        self.input_dim = torch.sqrt(args.random_dim)
        self.num_classes = 2

        # Data loading code
        kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
        x_train, y_train = generate_random_samples(n, args.random_dim)
        self.x_train = x_train
        self.y_train = y_train
        dataset = synthetic_dataset(x_train, y_train)
        self.train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)

