import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerTokenizer, LongformerModel
from sklearn.model_selection import train_test_split
import numpy as np

class ContrastiveDataset(Dataset):
    def __init__(self, df, tokenizer, col_name1, col_name2, label_col, max_length):
        self.df = df.reset_index(drop=True)
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.col_name1= col_name1
        self.col_name2 = col_name2
        self.label_col = label_col
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        resume_str = self.df[self.col_name1].iloc[idx]
        jd_str = self.df[self.col_name2].iloc[idx]
        label = self.df[self.label_col].iloc[idx]
        resume_encoding = self.tokenizer(resume_str, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        jd_encoding = self.tokenizer(jd_str, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        
        return {
            "resume_input_ids": resume_encoding["input_ids"].squeeze(0),
            "resume_attention_mask": resume_encoding["attention_mask"].squeeze(0),
            "jd_input_ids": jd_encoding["input_ids"].squeeze(0),
            "jd_attention_mask": jd_encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


def create_train_val_dataloaders(tokenizer, csv_path, batch_size, val_split, col_name1, col_name2, label_col, max_length, shuffle_train=True):
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=val_split, random_state=42)
    print(f"Training pairs: {len(train_df)}, Validation pairs: {len(val_df)}")
    
    train_dataset = ContrastiveDataset(train_df, tokenizer, col_name1, col_name2, label_col, max_length)
    val_dataset = ContrastiveDataset(val_df, tokenizer, col_name1, col_name2, label_col, max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader