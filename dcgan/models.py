import torch
import torch.nn as nn
from config import get_config

config = get_config()


class Generator(nn.Module):
    def __init__(self, n_gpus: int):
        super(Generator, self).__init__()
        self.n_gpus = n_gpus
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
        self.convT_1 = nn.ConvTranspose2d(
            in_channels=config["nz"],
            out_channels=config["ngf"] * 8,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn_1 = nn.BatchNorm2d(num_features=config["ngf"] * 8)
        self.convT_2 = nn.ConvTranspose2d(
            in_channels=config["ngf"] * 8,
            out_channels=config["ngf"] * 4,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn_2 = nn.BatchNorm2d(num_features=config["ngf"] * 4)
        self.convT_3 = nn.ConvTranspose2d(
            in_channels=config["ngf"] * 4,
            out_channels=config["ngf"] * 2,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn_3 = nn.BatchNorm2d(num_features=config["ngf"] * 2)
        self.convT_4 = nn.ConvTranspose2d(
            in_channels=config["ngf"] * 2,
            out_channels=config["ngf"],
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn_4 = nn.BatchNorm2d(num_features=config["ngf"])
        self.convT_5 = nn.ConvTranspose2d(
            in_channels=config["ngf"],
            out_channels=config["nc"],
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )

    def forward(self, x):
        x = self.relu(self.bn_1(self.convT_1(x)))
        x = self.relu(self.bn_2(self.convT_2(x)))
        x = self.relu(self.bn_3(self.convT_3(x)))
        x = self.relu(self.bn_4(self.convT_4(x)))
        x = self.tanh(self.convT_5(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, n_gpus):
        super(Discriminator, self).__init__()
        self.n_gpus = n_gpus
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(
            in_channels=config["nc"],
            out_channels=config["ndf"],
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=config["ndf"],
            out_channels=config["ndf"] * 2,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.conv3 = nn.Conv2d(
            in_channels=config["ndf"] * 2,
            out_channels=config["ndf"] * 4,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.conv4 = nn.Conv2d(
            in_channels=config["ndf"] * 4,
            out_channels=config["ndf"] * 8,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.conv5 = nn.Conv2d(
            in_channels=config["ndf"] * 8,
            out_channels=1,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn_1 = nn.BatchNorm2d(num_features=config["ndf"])
        self.bn_2 = nn.BatchNorm2d(num_features=config["ndf"] * 2)
        self.bn_3 = nn.BatchNorm2d(num_features=config["ndf"] * 4)
        self.bn_4 = nn.BatchNorm2d(num_features=config["ndf"] * 8)

    def forward(self, x):
        x = self.leaky_relu(self.bn_1(self.conv1(x)))
        x = self.leaky_relu(self.bn_2(self.conv2(x)))
        x = self.leaky_relu(self.bn_3(self.conv3(x)))
        x = self.leaky_relu(self.bn_4(self.conv4(x)))
        x = self.sigmoid(self.conv5(x))
        return x
