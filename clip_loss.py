import torch
import torch.nn as nn
import clip

class CLIPLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.model, _ = clip.load("ViT-B/32", device=device)
        self.device = device

    def forward(self, image, text_prompt):
        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(clip.tokenize([text_prompt]).to(self.device))
        loss = 1 - torch.cosine_similarity(image_features, text_features).mean()
        return loss
    
    