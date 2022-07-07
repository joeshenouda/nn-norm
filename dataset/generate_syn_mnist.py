import os
import copy
import torch
import random
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import argparse

def gen_labels(output_dim=10):
    torch.manual_seed(42)
    random.seed(42)

    mnist_labels = torch.randn(12000, output_dim)
    data_path = "data/"
    torch.save(mnist_labels, os.path.join(data_path,'2D_gaussian_labels_D_{}.pt'.format(output_dim)))

if __name__ == "__main__":
    gen_labels()