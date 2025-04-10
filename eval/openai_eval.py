import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from utils.loss import ContrastiveLoss
from openai import OpenAI
from google.colab import userdata
    
class OpenAI_Evaluator:
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
        
        self.api_key = userdata.get("openai")
        self.client = OpenAI(api_key=self.api_key)

    @staticmethod
    def _load_config(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)

    def _split_data(self, df, val_split):
        val_size = int(len(df) * val_split)
        return df.iloc[val_size:].reset_index(drop=True), df.iloc[:val_size].reset_index(drop=True)
    
    def _get_openai_embedding(self, text):
        res = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        ).data[0].embedding
        return res

    def evaluate(self):
        self.loss_fn.to(self.device)
        self.loss_fn.eval()

        total_loss = 0.0
        num_batches = 0
        batch_size = 16

        val_df = self.val_df

        for i in tqdm(range(0, len(val_df), batch_size), desc="OpenAI Evaluating"):
            batch = val_df.iloc[i:i+batch_size]

            anchor_1_texts = batch[self.col_name1].astype(str).tolist()
            anchor_2_texts = batch[self.col_name2].astype(str).tolist()
            labels = batch[self.label_col].astype(float).tolist()

            anchor_1_vecs = [self.get_openai_embedding(text) for text in anchor_1_texts]
            anchor_2_vecs = [self.get_openai_embedding(text) for text in anchor_2_texts]

            anchor_1_emb = torch.tensor(anchor_1_vecs, dtype=torch.float32).to(self.device)
            anchor_2_emb = torch.tensor(anchor_2_vecs, dtype=torch.float32).to(self.device)
            labels_tensor = torch.tensor(labels, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                loss = self.loss_fn(anchor_1_emb, anchor_2_emb, labels_tensor)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = round(total_loss / num_batches, 8)
        print(f"OpenAI-text-embedding-3-large Validation Contrastive Loss: {avg_loss}")
        return avg_loss