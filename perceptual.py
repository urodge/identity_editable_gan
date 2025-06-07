import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()

    def forward(self, input, target):
        input_feat = self.vgg(input)
        target_feat = self.vgg(target)
        return self.criterion(input_feat, target_feat)