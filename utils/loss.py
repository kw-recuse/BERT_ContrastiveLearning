import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerTokenizer, LongformerModel
from sklearn.model_selection import train_test_split
import numpy as np


class ContrastiveLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(ContrastiveLoss, self).__init__()
        self.loss_fn = nn.CosineEmbeddingLoss(reduction=reduction)
        
    def forward(self, resume_emb, jd_emb, labels):
        target = labels * 2 - 1 
        return self.loss_fn(resume_emb, jd_emb, target)


