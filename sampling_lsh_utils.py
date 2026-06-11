import sampling_utils
import torch
import torch.nn.functional as F
from transformers import GenerationConfig, StoppingCriteriaList, LogitsProcessor, LogitsProcessorList
from sbert_lsh_model import SBERTLSHModel
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
import numpy as np
from sampling_utils import SentenceEndCriteria, device
import hashlib
from nltk.tokenize import sent_tokenize 


class SemanticSeedLogitsProcessor(LogitsProcessor):
    # 👇 1. 初始化中加入 sweet_threshold 参数
    def __init__(self, vocab_size: int, gamma: float, delta: float, seed: int, sweet_threshold: float = 0.6):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.seed = seed
        self.sweet_threshold = sweet_threshold

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 👇 2. 计算当前 Logits 分布的香农熵
        # 先将 Logits 转为概率分布
        probs = torch.softmax(scores, dim=-1)
        # 计算熵：H = -sum(p * log(p))，加 1e-10 是为了防止对数里出现 0 导致 NaN
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        # 3. 决定绿名单 (原有逻辑)
        rng = torch.Generator(device='cpu')
        rng.manual_seed(self.seed)

        vocab_permutation = torch.randperm(self.vocab_size, generator=rng)
        greenlist_size = int(self.vocab_size * self.gamma)
        greenlist_ids = vocab_permutation[:greenlist_size].to(scores.device)

        # 👇 4. SWEET 门控过滤：仅对熵大于等于阈值的样本施加偏置
        for b in range(scores.size(0)):  # 遍历 batch 维度（通常生成时 batch size 为 1）
            if entropy[b] >= self.sweet_threshold:
                scores[b, greenlist_ids] += self.delta
                
        return scores

def get_seed_from_lsh(lsh_sig):
    sig_str = str(lsh_sig)
    return int(hashlib.md5(sig_str.encode('utf-8')).hexdigest(), 16) % (2**31 - 1)

def lsh_reject_completion(
        prompt: str,
        model: PreTrainedModel,  
        tokenizer: PreTrainedTokenizer, 
        gen_config: GenerationConfig,  
        lsh_model: SBERTLSHModel,  
        lsh_dim: int,  
        lmbd=0.5,
        device='cuda',
        margin=2.0, 
        sweet_threshold=0.6, # <--- 新增参数
        **kwargs
):
    delta = margin if margin > 0 else 2.0
    sent_end_criteria = SentenceEndCriteria(tokenizer)

    text_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    prompt_length = text_ids.size(1)
    full_text = prompt 

    while True:

        current_sents = sent_tokenize(full_text)
        
        if len(current_sents) == 0:
            context_sentence = ""
        elif len(current_sents) == 1:
            context_sentence = current_sents[0]
        else:
            if full_text.strip()[-1] in ['.', '!', '?', '"', "'", '”', '’']:
                context_sentence = current_sents[-1]
            else:
                context_sentence = current_sents[-2]

        lsh_sig = lsh_model.get_hash([context_sentence])[0]
        current_seed = get_seed_from_lsh(lsh_sig)

        # 👇 6. 在实例化 Processor 时把 sweet_threshold 传进去
        processor = SemanticSeedLogitsProcessor(len(tokenizer), lmbd, delta, current_seed, sweet_threshold)
        processors = LogitsProcessorList([processor])

        stopping_criteria = StoppingCriteriaList([sent_end_criteria])
        sent_end_criteria.update(full_text)

        new_text_ids = model.generate(
            text_ids,
            max_new_tokens=60, 
            min_new_tokens=3,
            do_sample=True,
            temperature=gen_config.temperature,
            repetition_penalty=gen_config.repetition_penalty,
            logits_processor=processors, 
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id
        )

        new_tokens = new_text_ids[0][text_ids.size(1):]
        new_sentence = tokenizer.decode(new_tokens, skip_special_tokens=True)

        if new_sentence.strip() == '':
            break

        full_text += new_sentence
        text_ids = new_text_ids

        if (text_ids.size(1) - prompt_length) >= gen_config.max_new_tokens - 1:
            break

    return full_text