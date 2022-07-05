import os
import copy
import torch
import random
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import argparse

torch.manual_seed(42)
random.seed(42)

mnist_labels = torch.randn(12000, 2)
data_path = "data/"
torch.save(mnist_labels, os.path.join(data_path,'2D_gaussian_labels.pt'))
