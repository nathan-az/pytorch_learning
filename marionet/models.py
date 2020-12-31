import copy
from torch import nn


class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        def calc_size_after_conv(start, kernel_size, stride):
            return (start - kernel_size) // stride + 1

        linear_size = calc_size_after_conv(
            calc_size_after_conv(calc_size_after_conv(84, 8, 4), 4, 2), 3, 1
        )

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(linear_size * linear_size * 64, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x, model_type):
        if model_type == "online":
            return self.online(x)
        elif model_type == "target":
            return self.target(x)
        else:
            raise ValueError(
                f"Expected model_type: 'online' or 'target', got: {model_type}"
            )
