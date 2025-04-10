from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from tqdm import tqdm
import pandas as pd
import json
import numpy as np

from utils.loss import ContrastiveLoss

class TFIDF_Evaluator:
    def __init__(self, config_file, **kwargs):
        self.config = self._load_config(config_file)

        for key in ['csv_file_path', 'col_name1', 'col_name2', 'label_col']:
            if key in kwargs:
                self.config[key] = kwargs[key]

        self.csv_file_path = self.config['csv_file_path']
        self.col_name1 = self.config['col_name1']
        self.col_name2 = self.config['col_name2']
        self.label_col = self.config['label_col']
        self.val_split = self.config['val_split']
        self.device = self.config['device']

        self.loss_fn = ContrastiveLoss()
        self.df = pd.read_csv(self.csv_file_path)
        self.train_df, self.val_df = self._split_data(self.df, self.val_split)

    @staticmethod
    def _load_config(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)

    def _split_data(self, df, val_split):
        val_size = int(len(df) * val_split)
        return df.iloc[val_size:].reset_index(drop=True), df.iloc[:val_size].reset_index(drop=True)

    def evaluate(self):
        self.loss_fn.to(self.device)
        self.loss_fn.eval()

        total_loss = 0.0
        num_batches = 0
        batch_size = 1

        val_df = self.val_df

        for i in tqdm(range(0, len(val_df), batch_size), desc="TF-IDF Evaluating"):
            batch = val_df.iloc[i:i+batch_size]

            anchor_1_text = batch[self.col_name1].values[0]
            anchor_2_text = batch[self.col_name2].values[0]
            label = float(batch[self.label_col].values[0])

            # Fit TF-IDF on just this pair
            vectorizer = TfidfVectorizer()
            vectorizer.fit([anchor_1_text, anchor_2_text])

            anchor_1_vec = vectorizer.transform([anchor_1_text]).toarray()
            anchor_2_vec = vectorizer.transform([anchor_2_text]).toarray()

            anchor_1_emb = torch.tensor(anchor_1_vec, dtype=torch.float32).to(self.device)
            anchor_2_emb = torch.tensor(anchor_2_vec, dtype=torch.float32).to(self.device)
            label_tensor = torch.tensor([label], dtype=torch.float32).to(self.device)

            with torch.no_grad():
                loss = self.loss_fn(anchor_1_emb, anchor_2_emb, label_tensor)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = round(total_loss / num_batches, 8)
        print(f"TF-IDF Validation Contrastive Loss: {avg_loss}")
        return avg_loss