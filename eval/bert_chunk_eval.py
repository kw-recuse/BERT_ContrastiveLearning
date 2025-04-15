import re
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import normalize
from utils.loss import ContrastiveLoss
import numpy as np
import faiss
import pandas as pd

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

        self.stop_ko = {"및", "등", "관련", "경험", "위한", "있는", "대한"}
        self.stop_en = {"and", "the", "for", "with", "of", "in", "to", "a", "an"}

    def preprocess(text, min_tokens):
        raw_paragraphs = re.split(r"\n{3,}", text.strip())
        return raw_paragraphs

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
    
    def topk_similarity(self, src_emb: np.ndarray, tgt_emb: np.ndarray, k: int = 3) -> float:
        dim = tgt_emb.shape[1]
        index = faiss.IndexFlatIP(dim)  
        index.add(tgt_emb)        
        k = min(k, tgt_emb.shape[0])    
        sims, _ = index.search(src_emb, k) 
        return float(sims.mean() + 1) / 2  
    
    def doc_similarity(self, resume: str, jd: str, k: int = 3, symmetric: bool = True) -> float:
        res_paras = self.preprocess(resume)
        jd_paras  = self.preprocess(jd)

        res_emb = self.embed_paragraphs(res_paras)
        jd_emb  = self.embed_paragraphs(jd_paras)
        sim_r2j = self.topk_similarity(res_emb, jd_emb, k)

        if not symmetric:
            return sim_r2j

        sim_j2r = self.topk_similarity(jd_emb, res_emb, k)
        return (sim_r2j + sim_j2r) / 2


    def evaluate(self):
        self.model.eval()

        df = pd.read_csv(self.csv_file_path)

        total_loss = 0.0
        num_rows = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc="ChunkBERT Evaluating"):
            resume = str(row[self.col_name1])
            jd     = str(row[self.col_name2])
            label  = float(row[self.label_col])

            # similarity in [0, 1]
            sim_score = self.doc_similarity(resume, jd, k=self.k)

            # compute MSE between sim and label
            sim_tensor = torch.tensor([sim_score], dtype=torch.float32, device=self.device)
            label_tensor = torch.tensor([label], dtype=torch.float32, device=self.device)

            mse = torch.nn.functional.mse_loss(sim_tensor, label_tensor)
            total_loss += mse.item()
            num_rows += 1
            
        avg_rmse = round(total_loss / num_rows, 8)
        print(f"ChunkBERT Validation Contrastive Loss: {avg_rmse}")
        return avg_rmse