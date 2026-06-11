import json
import random
import math
from dataclasses import dataclass
from typing import List
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup


# ================================================
# Config
# ================================================
@dataclass
class Config:
    train_path: str = "data.json"
    model_name: str = "/home/haojifei/dev_resource/huggingface/models/FacebookAI/roberta-base"
    lr: float = 2e-5
    epochs: int = 10
    batch_size: int = 16
    max_len: int = 128
    temperature: float = 0.07
    device: str = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()


# ================================================
# Utility: Random Masking
# ================================================
def random_mask_tokens(tokens: List[str], mask_rate: float, tokenizer):
    """随机mask部分token"""
    output = []
    for tok in tokens:
        if random.random() < mask_rate:
            output.append(tokenizer.mask_token)
        else:
            output.append(tok)
    return output


# ================================================
# Dataset
# ================================================
class ParaphraseDataset(Dataset):
    """
    每条数据格式：
    {
      "anchor": "...",
      "positives": ["...", "..."]
    }
    """

    def __init__(self, path):
        self.data = json.load(open(path, "r", encoding="utf-8"))
        self.tokenizer = RobertaTokenizer.from_pretrained(cfg.model_name)

    def __len__(self):
        return len(self.data)

    def encode_text(self, text):
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cfg.max_len
        )

    def augment_mask(self, text):
        """随机mask 5%、10%、15% 中的一种"""
        rates = [0.05, 0.10, 0.15]
        r = random.choice(rates)

        tokens = self.tokenizer.tokenize(text)
        masked = random_mask_tokens(tokens, mask_rate=r, tokenizer=self.tokenizer)
        masked_text = self.tokenizer.convert_tokens_to_string(masked)
        return masked_text

    def __getitem__(self, idx):
        item = self.data[idx]
        anchor = item["anchor"]
        positives = item["positives"]

        # 正例：任意采样一个 paraphrase
        pos = random.choice(positives)

        # 负例：masked 原句
        neg = self.augment_mask(anchor)

        return anchor, pos, neg


# ================================================
# SupCon model (Encoder + Projection Head)
# ================================================
class SupConModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(cfg.model_name)
        hidden = self.encoder.config.hidden_size  # 768

        # Projection head (2-layer MLP)
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        self.tokenizer = RobertaTokenizer.from_pretrained(cfg.model_name)

    def forward(self, texts):
        """输入list[str]，输出 L2-normalized 向量"""
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_len
        ).to(cfg.device)

        out = self.encoder(**enc).last_hidden_state[:, 0]  # CLS token

        z = self.proj(out)
        z = nn.functional.normalize(z, dim=-1)
        return z


# ================================================
# Supervised Contrastive Loss (公式6)
# ================================================
class SupConLoss(nn.Module):
    """论文公式6：multi-positive supervised contrastive"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: [N, d] 已归一化
        labels: [N]
        """
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)  # [N, 1]
        mask = torch.eq(labels, labels.T).float().to(device)  # [N, N]

        # Dot-product similarity
        sim = torch.div(torch.matmul(features, features.T), self.temperature)

        # 为避免正例包括自身，将对角置为极小值
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # 每行只包含与相同类的正例
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-9)

        loss = -mean_log_prob_pos.mean()
        return loss


# ================================================
# Train
# ================================================
def train():
    dataset = ParaphraseDataset(cfg.train_path)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = SupConModel().to(cfg.device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = SupConLoss(temperature=cfg.temperature)

    total_steps = len(loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    print("Total training samples:", len(dataset))

    for epoch in range(cfg.epochs):
        model.train()
        losses = []

        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}"):
            anchor_texts, pos_texts, neg_texts = batch

            # 合并：anchor, pos, neg
            texts = []
            labels = []

            # anchor = class 0, pos = class 0
            # neg = class 1
            for a, p, n in zip(anchor_texts, pos_texts, neg_texts):
                texts.append(a)
                labels.append(0)
                texts.append(p)
                labels.append(0)
                texts.append(n)
                labels.append(1)

            labels = torch.tensor(labels, dtype=torch.long).to(cfg.device)

            features = model(texts)  # [3B, 768]
            loss = loss_fn(features, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

        print(f"Epoch {epoch + 1}: loss={sum(losses) / len(losses):.4f}")

    # 保存模型
    torch.save(model.encoder.state_dict(), "encoder_finetuned.pt")
    print("Saved encoder_finetuned.pt")


if __name__ == "__main__":
    train()
