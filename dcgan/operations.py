import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def print_random_grid(data, device):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(data[0].to(device)[:64], padding=2, normalize=True).cpu(),
            (1, 2, 0),
        )
    )
