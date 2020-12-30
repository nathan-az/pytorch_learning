import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import random
from operations import Transition


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [None] * capacity
        self.total_ever = 0

    def push(self, *args):
        """
        Pushes s,a,r,s' to the memory. Cycles through the array (ring append)
        :param args:
        :return:
        """
        self.memory[self.total_ever % self.capacity] = Transition(*args)
        self.total_ever = self.total_ever + 1

    def __len__(self):
        """
        Returns the count of actual memories held in memory. Does not return the actual array length, as this is
        pre-allocated for efficiency.
        :return:
        """
        return min(self.total_ever, self.capacity)

    def sample(self, batch_size):
        return random.sample(self.memory[: len(self)], batch_size)


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - kernel_size) // stride + 1

        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        self.head = nn.Linear(conv_w * conv_h * 32, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x
