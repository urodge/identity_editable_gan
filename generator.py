import torch
import torch.nn as nn
from identity_editable_gan.models.unet_generator import UNetGenerator

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(base, base * 2, 4, 2, 1), nn.BatchNorm2d(base * 2), nn.ReLU()
        )
        self.middle = ResBlock(base * 2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1), nn.BatchNorm2d(base), nn.ReLU(),
            nn.ConvTranspose2d(base, out_channels, 4, 2, 1), nn.Tanh()
        )

    def forward(self, x):
        e = self.encoder(x)
        m = self.middle(e)
        d = self.decoder(m)
        return d
