import os
import copy
import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset


class CIFAR10:
    def __init__(self, args):
        """
        use args: num_workers, cuda, data_path, batch_size
        """
        super(CIFAR10, self).__init__()

        # use args:
        use_cuda = args.cuda
        num_workers = args.num_workers
        path = args.data_path
        batch_size = args.batch_size
        label_corruption = args.label_corruption  # float in [0, 1]

        # basic information
        self.input_dim = 32
        self.num_classes = 10
        self.input_channel = 3

        # Data loading code
        kwargs = {"num_workers": num_workers, "pin_memory": True} if use_cuda else {}
        transform = transforms.Compose(
                                [transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = datasets.CIFAR10(path, train=True, download=True, transform=transform)
        train_set, val_set = torch.utils.data.random_split(dataset, [55000, 5000])
        train_set = corrupt_this_subset(train_set, label_corruption) if label_corruption > 0 else train_set
        test_set = datasets.CIFAR10(path, train=False, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
        self.classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class CIFAR10_subset:
    def __init__(self, args):
        """
        use args: num_workers, cuda, data_path, batch_size
        """
        super(CIFAR10_subset, self).__init__()

        # use args:
        use_cuda = args.cuda
        num_workers = args.num_workers
        path = args.data_path
        batch_size = args.batch_size
        label_corruption = args.label_corruption  # float in [0, 1]

        # basic information
        self.input_dim = 32
        self.num_classes = 10
        self.input_channel = 3
        self.per_class_num = args.samps_per_class
        label_include = 10
        # in total per_class_num * label_include number of samples

        # Data loading code
        kwargs = {"num_workers": num_workers, "pin_memory": True} if use_cuda else {}
        transform = transforms.Compose(
                                [transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = datasets.CIFAR10(path, train=True, download=True, transform=transform)
        val_set = datasets.CIFAR10(path, train=True, download=True, transform=transform)

        # permute
        train_data = train_set.__dict__['data']
        train_targets = train_set.__dict__['targets']
        idx_rand = torch.randperm(60000)
        train_data = train_data[idx_rand]
        train_targets = train_targets[idx_rand]
        train_set.__dict__['data'] = train_data
        train_set.__dict__['targets'] = train_targets

        updated_train_data, updated_train_targets = [], []
        updated_val_data, updated_val_targets = [], []  # val set are those samples that are not included in the train
        count = torch.zeros(label_include)
        for i in range(60000):
            target_i = train_set.__dict__['targets'][i].item()
            sample_i = train_set.__dict__['data'][i]
            if target_i in torch.arange(label_include).tolist() and count[target_i] < self.per_class_num:
                updated_train_data.append(sample_i)
                updated_train_targets.append(target_i)
                count[target_i] += 1
            else:
                updated_val_data.append(sample_i)
                updated_val_targets.append(target_i)

        train_set.__dict__['targets'] = updated_train_targets
        train_set.__dict__['data'] = updated_train_data
        train_set = corrupt_this_dataset(train_set, label_corruption) if label_corruption > 0 else train_set
        val_set.__dict__['targets'] = updated_val_targets
        val_set.__dict__['data'] = updated_val_data
        # train_set = my_dataset(updated_train_data, updated_train_targets)
        # val_set = my_dataset(updated_val_data, updated_val_targets)

        test_set = datasets.MNIST(path, train=False, transform=transform)

        updated_test_data = []
        updated_test_targets = []
        for i in range(10000):
            target_i = test_set.__dict__['targets'][i].item()
            if target_i in torch.arange(label_include).tolist():
                sample_i = test_set.__dict__['data'][i]
                updated_test_data.append(sample_i)
                updated_test_targets.append(target_i)

        # test_set = my_dataset(updated_test_data, updated_test_targets)
        test_set.__dict__['targets'] = updated_test_targets
        test_set.__dict__['data'] = updated_test_data

        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)

