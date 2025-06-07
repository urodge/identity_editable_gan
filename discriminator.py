import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(base, base * 2, 4, 2, 1), nn.BatchNorm2d(base * 2), nn.LeakyReLU(0.2),
            nn.Conv2d(base * 2, 1, 4, 1, 0)
        )

    def forward(self, x):
        return self.net(x)
