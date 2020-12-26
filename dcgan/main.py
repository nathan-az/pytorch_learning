import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from config import get_config
from dcgan.models import Discriminator, Generator

config = get_config()
random.seed(config["seed"])
torch.manual_seed(config["seed"])

device = torch.device(
    "cuda:0" if (torch.cuda.is_available() and config["n_gpus"] > 0) else "cpu"
)
discriminator = Discriminator(config["n_gpus"]).to(device)
generator = Generator(config["n_gpus"]).to(device)

print(list(generator.children()))
