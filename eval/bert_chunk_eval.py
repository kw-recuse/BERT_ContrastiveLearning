import re
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import normalize

class ChunkBERT_Eval:
    def __init__(self, model_name="recuse/distiluse-base-multilingual-cased-v2-mean-dataV1", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.stop_ko = {"및", "등", "관련", "경험", "위한", "있는", "대한", "및"}
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

    def topk_similarity(self, src_emb, tgt_emb, k):
        dim = tgt_emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(tgt_emb)

        k = min(k, tgt_emb.shape[0])
        sims, _ = index.search(src_emb, k)

        return float(sims.mean() + 1) / 2

    def doc_similarity(self, resume, jd, k=3, symmetric=True):
        res_paras = self.preprocess(resume)
        jd_paras = self.preprocess(jd)

        res_emb = self.embed_paragraphs(res_paras)
        jd_emb = self.embed_paragraphs(jd_paras)

        sim_r2j = self.topk_similarity(res_emb, jd_emb, k)

        if not symmetric:
            return sim_r2j

        sim_j2r = self.topk_similarity(jd_emb, res_emb, k)

        return (sim_r2j + sim_j2r) / 2