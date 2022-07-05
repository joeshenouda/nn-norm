import os
import copy
import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset


class myMNISTDataset(Dataset):
    def __init__(self, tensor_data, tensor_label, transform=None):
        self.data = tensor_data  # [N, 28, 28, 1]
        self.label = tensor_label
        assert tensor_data.shape[0] == tensor_label.shape[0]
        self.transform = transform
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        sample = self.data[idx, :, :, 0]  # get rid of the extra dimension in [28, 28, 1]
        sample = Image.fromarray(sample.type(torch.uint8).numpy(), mode="L")
        label = self.label[idx]
        if self.transform:
            sample = self.transform(sample)[0]  # sample shape [1, 28, 28]
        return sample, label

class mySynthMNISTDataset(Dataset):
    def __init__(self, tensor_data):
        self.data = tensor_data
        tot_samps = self.data.shape[0]
        self.labels = torch.load('data/2D_gaussian_labels.pt')[:tot_samps]
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        return sample, label

class MNIST:
    def __init__(self, args):
        """
        use args: num_workers, cuda, data_path, batch_size
        """
        super(MNIST, self).__init__()

        # use args:
        use_cuda = args.cuda
        num_workers = args.num_workers
        path = args.data_path
        batch_size = args.batch_size
        label_corruption = args.label_corruption  # float in [0, 1]

        # basic information
        self.input_dim = 28
        self.num_classes = 10
        self.input_channel = 1

        # Data loading code
        kwargs = {"num_workers": num_workers, "pin_memory": True} if use_cuda else {}
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset = datasets.MNIST(path, train=True, download=True, transform=transform)
        train_set, val_set = torch.utils.data.random_split(dataset, [55000, 5000])
        train_set = corrupt_this_subset(train_set, label_corruption) if label_corruption > 0 else train_set
        test_set = datasets.MNIST(path, train=False, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)

class MNIST_subset:
    def __init__(self, args):
        """
        use args: num_workers, cuda, data_path, batch_size
        """
        super(MNIST_subset, self).__init__()

        # use args:
        use_cuda = args.cuda
        num_workers = args.num_workers
        path = args.data_path
        batch_size = args.batch_size
        label_corruption = args.label_corruption  # float in [0, 1]

        # basic information
        self.input_dim = 28
        self.num_classes = 10
        self.input_channel = 1
        self.per_class_num = args.samps_per_class
        label_include = 10
        # in total per_class_num * label_include number of samples

        # Data loading code
        kwargs = {"num_workers": num_workers, "pin_memory": True} if use_cuda else {}
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.MNIST(path, train=True, download=True, transform=transform)
        val_set = datasets.MNIST(path, train=True, download=True, transform=transform)

        # permute
        train_data = train_set.__dict__['data']
        train_targets = train_set.__dict__['targets']
        
        if args.load_pretrained_model:
            PATH_indx = os.path.join(args.results_dir, 'train_idxs.pt')
            idx_rand = torch.load(PATH_indx) if os.path.isfile(PATH_indx) else torch.randperm(60000)
            torch.save(idx_rand, PATH_indx)
        else:
            idx_rand = torch.randperm(60000)
            PATH_indx = os.path.join(args.dest_dir, "train_idxs.pt")
            torch.save(idx_rand, PATH_indx)
        
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



class MNIST_binary:
    def __init__(self, args):
        """
        use args: num_workers, cuda, data_path, batch_size

        Creates a dataset with just 0 and 1 MNIST samples for logistic loss binary shallow net
        """
        super(MNIST_binary, self).__init__()

        # use args:
        use_cuda = args.cuda
        num_workers = args.num_workers
        path = args.data_path
        batch_size = args.batch_size

        # basic information
        self.input_dim = 28
        self.num_classes = 2
        self.input_channel = 1
        self.per_class_num = args.samps_per_class
        label_include = 2

        # Data loading code
        kwargs = {"num_workers": num_workers, "pin_memory": True} if use_cuda else {}
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.MNIST(path, train=True, download=True, transform=transform)
        val_set = datasets.MNIST(path, train=True, download=True, transform=transform)

        # permute
        train_data = train_set.__dict__['data']
        train_targets = train_set.__dict__['targets']

        if args.load_pretrained_model:
            PATH_indx = os.path.join(args.results_dir, 'train_idxs.pt')
            idx_rand = torch.load(PATH_indx) if os.path.isfile(PATH_indx) else torch.randperm(60000)
            torch.save(idx_rand, PATH_indx)
        else:
            idx_rand = torch.randperm(60000)
            PATH_indx = os.path.join(args.dest_dir, "train_idxs.pt")
            torch.save(idx_rand, PATH_indx)
        
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

        #import ipdb; ipdb.set_trace()
        train_set.__dict__['targets'] = updated_train_targets
        train_set.__dict__['data'] = updated_train_data
        val_set.__dict__['targets'] = updated_val_targets
        val_set.__dict__['data'] = updated_val_data

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
   
class MNIST_binary_synth:
    def __init__(self, args):
        """
        use args: num_workers, cuda, data_path, batch_size

        Creates a dataset with just 0 and 1 MNIST samples where the labels are random gaussians

        I've left the train and validation sets untouched here since we are only interested in interpolating the training set.
        """
        super(MNIST_binary_synth, self).__init__()

        # use args:
        use_cuda = args.cuda
        num_workers = args.num_workers
        path = args.data_path
        batch_size = args.batch_size

        # basic information
        self.input_dim = 28
        self.num_classes = 2
        self.input_channel = 1
        self.per_class_num = args.samps_per_class
        label_include = 2

        # Data loading code
        kwargs = {"num_workers": num_workers, "pin_memory": True} if use_cuda else {}
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.MNIST(path, train=True, download=True, transform=transform)
        val_set = datasets.MNIST(path, train=True, download=True, transform=transform)

        train_data = train_set.__dict__['data']
        train_targets = train_set.__dict__['targets']

        if args.load_pretrained_model:
            PATH_indx = os.path.join(args.results_dir, 'train_idxs.pt')
            idx_rand = torch.load(PATH_indx) if os.path.isfile(PATH_indx) else torch.randperm(60000)
            torch.save(idx_rand, PATH_indx)
        else:
            idx_rand = torch.randperm(60000)
            PATH_indx = os.path.join(args.dest_dir, "train_idxs.pt")
            torch.save(idx_rand, PATH_indx)
        
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
        #import ipdb; ipdb.set_trace()
        
        upd_train_data_unsq = [torch.unsqueeze(a,0) for a in updated_train_data]
        updated_train_data_tens = torch.cat(upd_train_data_unsq)
        train_set = mySynthMNISTDataset(updated_train_data_tens)
        
        val_set.__dict__['targets'] = updated_val_targets
        val_set.__dict__['data'] = updated_val_data


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
