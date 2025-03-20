import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(ContrastiveLoss, self).__init__()
        self.reduction = reduction

    def forward(self, resume_emb, jd_emb, labels):
        cos_sim = nn.functional.cosine_similarity(resume_emb, jd_emb)
        scaled_cos_sim = (cos_sim + 1) / 2
        loss = nn.functional.mse_loss(scaled_cos_sim, labels, reduction=self.reduction)
        return loss
