from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from tqdm import tqdm
import json
from utils.loss import ContrastiveLoss
from data.dataloader import create_train_val_dataloaders

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
        self.batch_size = 1  # required since TF-IDF vector dims vary
        self.max_length = self.config.get('max_length', 512)  # fallback

        self.loss_fn = ContrastiveLoss()

        # use the same dataloader creation logic as BERT_Evaluator
        _, self.val_dataloader = create_train_val_dataloaders(
            tokenizer=None,  # TF-IDF doesn't use tokenizer
            csv_file_path=self.csv_file_path,
            batch_size=self.batch_size,
            val_split=self.val_split,
            col_name1=self.col_name1,
            col_name2=self.col_name2,
            label_col=self.label_col,
            max_length=self.max_length
        )

    @staticmethod
    def _load_config(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)

    def evaluate(self):
        self.loss_fn.to(self.device)
        self.loss_fn.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_dataloader, desc="TF-IDF Evaluating"):
            resume_text = batch[self.col_name1][0]
            jd_text = batch[self.col_name2][0]
            label = float(batch["label"][0])

            vectorizer = TfidfVectorizer()
            vectorizer.fit([resume_text, jd_text])

            resume_vec = vectorizer.transform([resume_text]).toarray()
            jd_vec = vectorizer.transform([jd_text]).toarray()

            resume_emb = torch.tensor(resume_vec, dtype=torch.float32).to(self.device)
            jd_emb = torch.tensor(jd_vec, dtype=torch.float32).to(self.device)
            label_tensor = torch.tensor([label], dtype=torch.float32).to(self.device)

            with torch.no_grad():
                loss = self.loss_fn(resume_emb, jd_emb, label_tensor)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = round(total_loss / num_batches, 8)
        print(f"TF-IDF Validation Contrastive Loss: {avg_loss}")
        return avg_loss
