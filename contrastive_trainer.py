import json
import random
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from datasets import load_from_disk
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer
from transformers.utils.versions import require_version
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from peft import LoraConfig, get_peft_model, TaskType
from paraphrase_gen import random_mask_tokens

torch.autograd.set_detect_anomaly(True)

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)

logger = logging.getLogger(__name__)


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)  # [N, 1]
        mask = torch.eq(labels, labels.T).float().to(device)  # [N, N]

        sim = torch.div(torch.matmul(features, features.T), self.temperature)

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-9)

        loss = -mean_log_prob_pos.mean()
        return loss



class SupConModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        encoder = AutoModel.from_pretrained(cfg.model_name, trust_remote_code=True)
        
        encoder.gradient_checkpointing_enable()
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )
        self.encoder = get_peft_model(encoder, peft_config)
        self.encoder.print_trainable_parameters() 
        
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        z = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        z = torch.nn.functional.normalize(z, p=2, dim=1)
        return z

def encode(attention_mask, model_output=None, token_embeddings=None):
    if model_output != None:
        embedding = mean_pooling(attention_mask, model_output=model_output)
    else:
        embedding = mean_pooling(
            attention_mask, token_embeddings=token_embeddings)
    return embedding


def mean_pooling(attention_mask, model_output=None, token_embeddings=None):
    if (model_output != None and token_embeddings == None):
        token_embeddings = model_output[0].to("cuda") 
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def cosine_distance_matrix(x, y):
    return F.cosine_similarity(
        x.view(x.size(0), 1, x.size(1))
        .expand(x.size(0), y.size(0), x.size(1))
        .contiguous()
        .view(-1, x.size(1)),
        y.expand(x.size(0), y.size(0), y.size(1)).flatten(end_dim=1),
    ).view(x.size(0), y.size(0))


class ParaphraseDataset(Dataset):
    def __init__(self, path, cfg):
        self.config = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        hf_dataset = load_from_disk(path)
        self.data = hf_dataset['train'] if 'train' in hf_dataset else hf_dataset 
        self.is_hf = True
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        anchor = item['text']
        # 提取所有可用的洗稿文本
        positives = [
            item.get(f'para_text{i}', '') for i in range(1, 9)
        ]
        
        if isinstance(anchor, list):
            anchor = " ".join(anchor)
        positives = [" ".join(p) if isinstance(p, list) else p for p in positives if p]
        
        pos = random.choice(positives)
        return anchor, pos

    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.is_hf:
            anchor = item['text']
            # 提取所有可用的洗稿文本
            positives = [
                item.get(f'para_text{i}', '') for i in range(1, 9)
            ]
            
            if isinstance(anchor, list):
                anchor = " ".join(anchor)
            positives = [" ".join(p) if isinstance(p, list) else p for p in positives if p]
            pos = random.choice([p for p in positives if len(p.strip()) > 0]) 
        else:
            anchor = item["anchor"]
            positives = item["positives"]
            pos = random.choice(positives)

        return anchor, pos

class ParaphraseContrastiveTrainer(Trainer):
    def __init__(self, delta, *args, **kwargs):
        self.inaccurate = 0
        self.total = 0
        self.delta = delta
        self.sim_loss = nn.MarginRankingLoss(
            margin=self.delta, reduction='sum')
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        if return_outputs:
            raise NotImplementedError

        text_outputs = model(
            input_ids=inputs["text_input_ids"],
            attention_mask=inputs["text_attention_mask"],
        )
        text_embeddings = mean_pooling(
            model_output=text_outputs, attention_mask=inputs["text_attention_mask"])

        para_outputs = model(
            input_ids=inputs["para_input_ids"],
            attention_mask=inputs["para_attention_mask"],
        )
        para_embeddings = mean_pooling(
            model_output=para_outputs, attention_mask=inputs["para_attention_mask"])

        sims = cosine_distance_matrix(text_embeddings, para_embeddings)

        pos = sims.diag()
        n = sims.size(0)
        mask = torch.ones(n, n, device=sims.device) - torch.eye(n, device=sims.device)
        masked_sims = sims * mask - torch.eye(n, device=sims.device)
        neg = masked_sims.max(dim=1).values
        loss = self.sim_loss(pos, neg, torch.ones_like(pos))
        inaccurate = 0
        for i in range(n):
            if torch.argmax(sims[i]) != i:
                self.inaccurate += 1
        self.total += n
        print(f"grand contrastive acc: {1 - self.inaccurate / self.total}")

        return (loss, text_outputs) if return_outputs else loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            loss = self.compute_loss(model, inputs)
        return (loss, None, None)
