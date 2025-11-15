import json
import random
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer
from transformers.utils.versions import require_version
from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup

from paraphrase_gen import random_mask_tokens

torch.autograd.set_detect_anomaly(True)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.28.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)

logger = logging.getLogger(__name__)


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


class SupConModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
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
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_len
        ).to(self.config.device)

        out = self.encoder(**enc).last_hidden_state[:, 0]  # CLS token

        z = self.proj(out)
        z = nn.functional.normalize(z, dim=-1)
        return z


def encode(attention_mask, model_output=None, token_embeddings=None):
    if model_output != None:
        embedding = mean_pooling(attention_mask, model_output=model_output)
    else:
        embedding = mean_pooling(
            attention_mask, token_embeddings=token_embeddings)
    # sentence_embeddings = F.normalize(embedding, p=2, dim=1)
    return embedding


def mean_pooling(attention_mask, model_output=None, token_embeddings=None):
    if (model_output != None and token_embeddings == None):
        token_embeddings = model_output[
            0
        ].to("cuda")  # First element of model_output contains all token embeddings
        # print(f"token_embeddings: {token_embeddings.size()}")
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
    """
    每条数据格式：
    {
      "anchor": "...",
      "positives": ["...", "..."]
    }
    """

    def __init__(self, path, cfg):
        self.config = cfg
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
            max_length=self.config.max_len
        )

    def augment_mask(self, text):
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
        pos = random.choice(positives)
        neg = self.augment_mask(anchor)
        return anchor, pos, neg


class ParaphraseContrastiveTrainer(Trainer):
    def __init__(self, delta, *args, **kwargs):
        self.inaccurate = 0
        self.total = 0
        self.delta = delta
        self.sim_loss = nn.MarginRankingLoss(
            margin=self.delta, reduction='sum')
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
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
        # neg = off_diag(sims).max(dim=1).values
        n = sims.size(0)
        mask = torch.ones(n, n, device=sims.device) - torch.eye(n, device=sims.device)
        masked_sims = sims * mask - torch.eye(n, device=sims.device)
        # change to dim = 0 and see if there's any difference
        neg = masked_sims.max(dim=1).values
        # use max cos sim in batch as negative example
        loss = self.sim_loss(pos, neg, torch.ones_like(pos))
        inaccurate = 0
        for i in range(n):
            if torch.argmax(sims[i]) != i:
                self.inaccurate += 1
        self.total += n
        print(f"grand contrastive acc: {1 - self.inaccurate / self.total}")
        # print("---------------------------------")

        return (loss, text_outputs) if return_outputs else loss

    # import this to make sure the trainer calls the right loss func during eval and prediction
    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and ovferride to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        with torch.no_grad():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            loss = self.compute_loss(model, inputs)
        return (loss, None, None)
