import os 
import json
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as F
from models.base import load_tokenizer_and_model
from data.dataloader import create_train_val_dataloaders
from utils.loss import ContrastiveLoss

class Trainer:
    def __init__(self, config_file, **kwargs):
        self.config = self._load_config(config_file)
        
        for key in ['checkpoints_path', 'csv_file_path', 'col_name1', 'col_name2', 'label_col']:
            if key in kwargs:
                self.config[key] = kwargs[key]

        self.col_name1 = self.config['col_name1']
        self.col_name2 = self.config['col_name2']
        self.label_col = self.config['label_col']
        self.model_name = self.config['model_name']
        self.device = self.config['device']
        self.batch_size = self.config['batch_size']
        self.val_split = self.config['val_split']
        self.checkpoints_path = self.config['checkpoints_path']
        self.lr = self.config['lr']
        self.epoch_num = self.config['epoch_num']
        self.csv_file_path = self.config['csv_file_path']
        self.num_logs_per_epoch = self.config['num_logs_per_epoch']
        self.use_fp16 = self.config['use_fp16']
        self.patience = self.config['patience']
        self.max_length = self.config['max_length']
        self.embedding_option = self.config['embedding_option']
        
        self.patience_counter = 0
        self.best_val_loss = 10000.0
        
        # make checjpoint dir
        os.makedirs(self.checkpoints_path, exist_ok=True)
        
        # load model and tokenizer
        self.tokenizer, self.model = load_tokenizer_and_model(self.model_name)
        self.model = self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.train_dataloader, self.val_dataloader = create_train_val_dataloaders(self.tokenizer, self.csv_file_path, self.batch_size, self.val_split, self.col_name1, self.col_name2, self.label_col, self.max_length)
        self.scaler = GradScaler()
        self.loss_fn = ContrastiveLoss()
        self.log_step = len(self.train_dataloader) // self.num_logs_per_epoch
        
        self.train_losses = []
        self.val_losses = []
        
    @staticmethod
    def _load_config(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
        
    def save_checkpoint(self, epoch, step, val_loss):
        checkpoint_path = os.path.join(self.checkpoints_path, f"epoch{epoch}_step{step}_loss{val_loss}.pt")
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict()
        }, checkpoint_path)
        tqdm.write(f"Checkpoint saved at {checkpoint_path}")
    
    
    def one_step_forward(self, anchor_1_input_ids, anchor_1_attention_mask, anchor_2_input_ids, anchor_2_attention_mask):
        anchor_1_outputs = self.model(anchor_1_input_ids, attention_mask=anchor_1_attention_mask)
        anchor_2_outputs = self.model(anchor_2_input_ids, attention_mask=anchor_2_attention_mask)
        
        if self.embedding_option == "CLS":
            anchor_1_emb = anchor_1_outputs.last_hidden_state[:, 0, :]
            anchor_2_emb = anchor_2_outputs.last_hidden_state[:, 0, :]
        else:
            # self.embedding_option == "Mean":
            anchor_1_last_hidden = anchor_1_outputs.last_hidden_state # (N, T, hidden_size)
            anchor_2_last_hidden = anchor_2_outputs.last_hidden_state # (N, T, hidden_size)
            
            anchor_1_mask = anchor_1_attention_mask.unsqueeze(-1).expand_as(anchor_1_last_hidden)  # (N, T, hidden_size)
            anchor_2_mask = anchor_2_attention_mask.unsqueeze(-1).expand_as(anchor_2_last_hidden) # (N, T, hidden_size)
            
            epsilon = 1e-8
            anchor_1_emb = (anchor_1_last_hidden * anchor_1_mask).sum(dim=1) / (anchor_1_mask.sum(dim=1) + epsilon)  # (N, hidden_size)
            anchor_2_emb = (anchor_2_last_hidden * anchor_2_mask).sum(dim=1) / (anchor_2_mask.sum(dim=1) + epsilon) # (N, hidden_size)
            
        return anchor_1_emb, anchor_2_emb
        
        
    def evaluate_val_loss(self, epoch, step):
        self.model.eval()
        total_val_loss = 0.0
        num_batchs = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                resume_input_ids = batch["resume_input_ids"].to(self.device)
                resume_attention_mask = batch["resume_attention_mask"].to(self.device)
                jd_input_ids = batch["jd_input_ids"].to(self.device)
                jd_attention_mask = batch["jd_attention_mask"].to(self.device)
                labels = batch["label"].to(self.device).float()
                
                if self.use_fp16:
                    with autocast():
                        resume_emb, jd_emb = self.one_step_forward(resume_input_ids, resume_attention_mask, jd_input_ids, jd_attention_mask)
                        loss = self.loss_fn(resume_emb, jd_emb, labels)
                else:
                    resume_emb, jd_emb = self.one_step_forward(resume_input_ids, resume_attention_mask, jd_input_ids, jd_attention_mask)
                    loss = self.loss_fn(resume_emb, jd_emb, labels)
                    
                total_val_loss += loss.item()
                num_batchs += 1
                
        avg_val_loss = total_val_loss / num_batchs
        avg_val_loss = round(avg_val_loss, 8)
        self.val_losses.append(avg_val_loss)
        print(f"Epoch {epoch}, Step {step} - Avg Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss
        
    def train(self):
        self.model.train()
        for epoch in range(self.epoch_num):
            progress_bar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), desc=f"Epoch {epoch+1}", position=0, leave=True)
            for step, batch in progress_bar:
                resume_input_ids = batch["resume_input_ids"].to(self.device)
                resume_attention_mask = batch["resume_attention_mask"].to(self.device)
                jd_input_ids = batch["jd_input_ids"].to(self.device)
                jd_attention_mask = batch["jd_attention_mask"].to(self.device)
                labels = batch["label"].to(self.device).float()

                self.optimizer.zero_grad()
                
                if self.use_fp16:
                    with autocast():
                        resume_emb, jd_emb = self.one_step_forward(resume_input_ids, resume_attention_mask, jd_input_ids, jd_attention_mask)
                        loss = self.loss_fn(resume_emb, jd_emb, labels)   
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    resume_emb, jd_emb = self.one_step_forward(resume_input_ids, resume_attention_mask, jd_input_ids, jd_attention_mask)
                    loss = self.loss_fn(resume_emb, jd_emb, labels)
                    loss.backward()
                    self.optimizer.step()
                    
                self.train_losses.append(round(loss.item(), 8))
                progress_bar.set_postfix(Step=step+1, Loss=round(loss.item(), 8))
                    
                # log step
                if (step+1) % self.log_step == 0 or step == len(self.train_dataloader) - 1:
                    val_loss = self.evaluate_val_loss(epoch+1, step+1)
                    self.save_checkpoint(epoch+1, step+1, val_loss)
                    
                    
                    # early stopping
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.patience:
                            return # stop the training