import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, semi_negative, negative):
        pos_dist = F.cosine_similarity(anchor, positive)
        semi_neg_dist = F.cosine_similarity(anchor, semi_negative)
        neg_dist = F.cosine_similarity(anchor, negative)

        loss = torch.relu(pos_dist - semi_neg_dist + self.margin) + torch.relu(semi_neg_dist - neg_dist + self.margin)
        return loss.mean()
