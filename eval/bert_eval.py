import os
import json
import torch
from torch.cuda.amp import autocast
from models.base import load_tokenizer_and_model
from data.dataloader import create_train_val_dataloaders
from utils.loss import ContrastiveLoss
from tqdm import tqdm

class BERT_Evaluator:
    def __init__(self, config_file, **kwargs):
        self.config = self._load_config(config_file)

        for key in ['csv_file_path', 'col_name1', 'col_name2', 'label_col']:
            if key in kwargs:
                self.config[key] = kwargs[key]

        self.col_name1 = self.config['col_name1']
        self.col_name2 = self.config['col_name2']
        self.label_col = self.config['label_col']
        self.model_name = self.config['model_name']
        self.device = self.config['device']
        self.batch_size = self.config['batch_size']
        self.val_split = self.config['val_split']
        self.csv_file_path = self.config['csv_file_path']
        self.use_fp16 = self.config['use_fp16']
        self.max_length = self.config['max_length']
        self.embedding_option = self.config['embedding_option']

        # load mode and tokenizer
        self.tokenizer, self.model = load_tokenizer_and_model(self.model_name)
        self.model.to(self.device)

        # val dataloaders
        _, self.val_dataloader = create_train_val_dataloaders(
            self.tokenizer,
            self.csv_file_path,
            self.batch_size,
            self.val_split,
            self.col_name1,
            self.col_name2,
            self.label_col,
            self.max_length
        )

        self.loss_fn = ContrastiveLoss()

    @staticmethod
    def _load_config(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)

    def _one_step_forward(self, anchor_1_input_ids, anchor_1_attention_mask, anchor_2_input_ids, anchor_2_attention_mask):
        anchor_1_outputs = self.model(anchor_1_input_ids, attention_mask=anchor_1_attention_mask)
        anchor_2_outputs = self.model(anchor_2_input_ids, attention_mask=anchor_2_attention_mask)

        if self.embedding_option == "CLS":
            anchor_1_emb = anchor_1_outputs.last_hidden_state[:, 0, :]
            anchor_2_emb = anchor_2_outputs.last_hidden_state[:, 0, :]
        else:  
            # mean pooling
            anchor_1_last_hidden = anchor_1_outputs.last_hidden_state
            anchor_2_last_hidden = anchor_2_outputs.last_hidden_state

            anchor_1_mask = anchor_1_attention_mask.unsqueeze(-1).expand_as(anchor_1_last_hidden)
            anchor_2_mask = anchor_2_attention_mask.unsqueeze(-1).expand_as(anchor_2_last_hidden)

            epsilon = 1e-8
            anchor_1_emb = (anchor_1_last_hidden * anchor_1_mask).sum(dim=1) / (anchor_1_mask.sum(dim=1) + epsilon)
            anchor_2_emb = (anchor_2_last_hidden * anchor_2_mask).sum(dim=1) / (anchor_2_mask.sum(dim=1) + epsilon)

        return anchor_1_emb, anchor_2_emb

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="BERT Evaluating", leave=False):
                resume_input_ids = batch["resume_input_ids"].to(self.device)
                resume_attention_mask = batch["resume_attention_mask"].to(self.device)
                jd_input_ids = batch["jd_input_ids"].to(self.device)
                jd_attention_mask = batch["jd_attention_mask"].to(self.device)
                labels = batch["label"].to(self.device).float()

                if self.use_fp16:
                    with autocast():
                        resume_emb, jd_emb = self._one_step_forward(resume_input_ids, resume_attention_mask, jd_input_ids, jd_attention_mask)
                        loss = self.loss_fn(resume_emb, jd_emb, labels)
                else:
                    resume_emb, jd_emb = self._one_step_forward(resume_input_ids, resume_attention_mask, jd_input_ids, jd_attention_mask)
                    loss = self.loss_fn(resume_emb, jd_emb, labels)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = round(total_loss / num_batches, 8)
        print(f"BERT Validation Contrastive Loss: {avg_loss}")
        return avg_loss