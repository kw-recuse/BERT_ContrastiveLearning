import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import train_test_split
import numpy as np

class ContrastiveDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=2048):
        self.df = df.reset_index(drop=True)
        self.max_length = max_length
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        resume_str = self.df["resume_str"].iloc[idx]
        jd_str = self.df["jd_str"].iloc[idx]
        label = self.df["label"].iloc[idx]
        resume_encoding = self.tokenizer(resume_str, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        jd_encoding = self.tokenizer(jd_str, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        
        return {
            "resume_input_ids": resume_encoding["input_ids"].squeeze(0),
            "resume_attention_mask": resume_encoding["attention_mask"].squeeze(0),
            "jd_input_ids": jd_encoding["input_ids"].squeeze(0),
            "jd_attention_mask": jd_encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float) 
        }

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size=4):
        self.labels = labels
        self.batch_size = batch_size
        
        self.indices_0 = np.where(labels == 0)[0].tolist()
        self.indices_05 = np.where(labels == 0.5)[0].tolist()
        self.indices_1 = np.where(labels == 1)[0].tolist()

        self.num_batches = min(
            len(self.indices_0) // 2, 
            len(self.indices_05),     
            len(self.indices_1)     
        )
    
    def __iter__(self):
        # Shuffle indices for each label
        np.random.shuffle(self.indices_0)
        np.random.shuffle(self.indices_05)
        np.random.shuffle(self.indices_1)
        
        # Generate batches
        for i in range(self.num_batches):
            batch = (
                self.indices_0[i * 2 : (i + 1) * 2] +  # Two '0's
                [self.indices_05[i]] +                 # One '0.5'
                [self.indices_1[i]]                    # One '1'
            )
            yield batch
    
    def __len__(self):
        return self.num_batches

def create_train_val_dataloaders(tokenizer, csv_path, batch_size, val_split):
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_df, val_df = train_test_split(df, test_size=val_split, random_state=42)
    print(f"Training pairs: {len(train_df)}, Validation pairs: {len(val_df)}")
    
    train_dataset = ContrastiveDataset(train_df, tokenizer)
    train_labels = train_df["label"].values 
    train_sampler = BalancedBatchSampler(train_labels, batch_size=batch_size)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False 
    )
    
    val_dataset = ContrastiveDataset(val_df, tokenizer)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_dataloader, val_dataloader