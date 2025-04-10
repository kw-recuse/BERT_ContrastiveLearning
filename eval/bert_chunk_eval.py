import re
import json
import torch
import faiss
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import normalize
from utils.loss import ContrastiveLoss
from data.dataloader import create_train_val_dataloaders


class ChunkBERT_Eval:
    def __init__(self, config_file, csv_file_path, col_name1, col_name2, label_col, k):
        with open(config_file, 'r') as f:
            self.config = json.load(f)

        # Set file paths and column names
        self.csv_file_path = csv_file_path
        self.col_name1 = col_name1
        self.col_name2 = col_name2
        self.label_col = label_col
        self.k = k 

        self.model_name = self.config["model_name"]
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = self.config["batch_size"]
        self.val_split = self.config["val_split"]
        self.max_length = self.config["max_length"]
        self.embedding_option = self.config.get("embedding_option", "mean")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        self.loss_fn = ContrastiveLoss().to(self.device)

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

        self.stop_ko = {"및", "등", "관련", "경험", "위한", "있는", "대한"}
        self.stop_en = {"and", "the", "for", "with", "of", "in", "to", "a", "an"}

    def preprocess(self, text, min_tokens=3):
        raw_paragraphs = re.split(r"(?:\n{2,}|[•\-]\s*)", text)
        paragraphs = []

        for p in raw_paragraphs:
            tokens = re.findall(r"[가-힣A-Za-z0-9\+#\.]+", p.lower())
            tokens = [t for t in tokens if t not in self.stop_ko and t not in self.stop_en]
            if len(tokens) >= min_tokens:
                paragraphs.append(" ".join(tokens))

        return paragraphs

    @torch.inference_mode()
    def embed_paragraphs(self, paragraphs, batch_size=16):
        all_embs = []

        for i in range(0, len(paragraphs), batch_size):
            batch = paragraphs[i : i + batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)

            output = self.model(**encoded)
            token_emb = output.last_hidden_state

            mask = encoded["attention_mask"].unsqueeze(-1).float()
            sent_emb = (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            sent_emb = normalize(sent_emb, p=2, dim=1)

            all_embs.append(sent_emb.cpu())

        return torch.cat(all_embs).numpy()

    def evaluate(self):
        self.model.eval()
        self.loss_fn.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="ChunkBERT Evaluating"):
                resume_input_ids = batch["resume_input_ids"].to(self.device)
                resume_attention_mask = batch["resume_attention_mask"].to(self.device)
                jd_input_ids = batch["jd_input_ids"].to(self.device)
                jd_attention_mask = batch["jd_attention_mask"].to(self.device)
                labels = batch["label"].to(self.device).float()

                resume_outputs = self.model(resume_input_ids, attention_mask=resume_attention_mask)
                jd_outputs = self.model(jd_input_ids, attention_mask=jd_attention_mask)

                def mean_pool(last_hidden, mask):
                    mask = mask.unsqueeze(-1).float()
                    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

                resume_vec = mean_pool(resume_outputs.last_hidden_state, resume_attention_mask)
                jd_vec = mean_pool(jd_outputs.last_hidden_state, jd_attention_mask)

                loss = self.loss_fn(resume_vec, jd_vec, labels)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = round(total_loss / num_batches, 8)
        print(f"ChunkBERT Validation Contrastive Loss: {avg_loss}")
        return avg_loss
