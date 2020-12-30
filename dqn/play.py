import gym
import math
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

from models import DQN, Transition

env = gym.make("CartPole-v0").unwrapped
plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# using s, a, r, s' convention
