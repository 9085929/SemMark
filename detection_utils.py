from sklearn.metrics import roc_curve, auc
import sampling_utils
import numpy as np
import torch
import math
import hashlib
import matplotlib.pyplot as plt
import os
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import sent_tokenize
device = "cuda" if torch.cuda.is_available() else "cpu"
rng = torch.Generator(device)

orig_model_from_pretrained = AutoModel.from_pretrained
orig_tok_from_pretrained = AutoTokenizer.from_pretrained

def my_model_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
    if pretrained_model_name_or_path == "microsoft/deberta-xlarge-mnli":
        pretrained_model_name_or_path = "/home/haojifei/dev_resource/huggingface/models/microsoft/deberta-xlarge-mnli"
    return orig_model_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

def my_tok_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
    if pretrained_model_name_or_path == "microsoft/deberta-xlarge-mnli":
        pretrained_model_name_or_path = "/home/haojifei/dev_resource/huggingface/models/microsoft/deberta-xlarge-mnli"
    return orig_tok_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

AutoModel.from_pretrained = classmethod(my_model_from_pretrained)
AutoTokenizer.from_pretrained = classmethod(my_tok_from_pretrained)


def get_seed_from_lsh(lsh_sig):
    sig_str = str(lsh_sig)
    return int(hashlib.md5(sig_str.encode('utf-8')).hexdigest(), 16) % (2**31 - 1)

def detect_lsh(raw_text, lsh_model, lmbd, lsh_dim, tokenizer):
    import math
    import torch
    from nltk.tokenize import sent_tokenize
    from detection_utils import get_seed_from_lsh
    
    if isinstance(raw_text, list):
        raw_text = " ".join(raw_text)
    prompt = extract_prompt_from_text(raw_text, len_prompt=32)
    generated_text = raw_text[len(prompt):].strip()
    if not generated_text:
        return None

    gen_sents = sent_tokenize(generated_text)
    
    if len(gen_sents) > 1 and gen_sents[-1].strip()[-1] not in ".!?。！？\"'":
        gen_sents = gen_sents[:-1]

    if len(gen_sents) == 0:
        return None

    chunks = [prompt] + gen_sents

    total_green = 0
    total_tokens = 0
    vocab_size = len(tokenizer)

    last_end_pos = raw_text.find(chunks[0])
    if last_end_pos != -1:
        last_end_pos += len(chunks[0])
    else:
        last_end_pos = 0

    for i in range(1, len(chunks)):
        context_sentence = chunks[i-1]
        current_sentence = chunks[i]
        
        curr_start = raw_text.find(current_sentence, last_end_pos)
        if curr_start == -1:
            chunk_to_encode = current_sentence
            old_last_end_pos = last_end_pos
            chunk_tokens = tokenizer.encode(chunk_to_encode, add_special_tokens=False)
        else:
            curr_end = curr_start + len(current_sentence)
            chunk_to_encode = raw_text[last_end_pos : curr_end]
            old_last_end_pos = last_end_pos
            last_end_pos = curr_end
            
            full_text = raw_text[:curr_end] 
            context_text = raw_text[:old_last_end_pos]
            
            full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
            context_tokens = tokenizer.encode(context_text, add_special_tokens=False)
            chunk_tokens = full_tokens[len(context_tokens):]
            
        lsh_sig = lsh_model.get_hash([context_sentence])[0]
        current_seed = get_seed_from_lsh(lsh_sig)

        rng = torch.Generator(device='cpu')
        rng.manual_seed(current_seed)
        vocab_permutation = torch.randperm(vocab_size, generator=rng)
        greenlist = set(vocab_permutation[:int(vocab_size * lmbd)].tolist())

        for tok in chunk_tokens:
            if tok in greenlist:
                total_green += 1
            total_tokens += 1

    if total_tokens == 0:
        return None

    expected_green = lmbd * total_tokens
    variance = total_tokens * lmbd * (1 - lmbd)

    if variance == 0:
        return None

    zscore = (total_green - expected_green) / math.sqrt(variance)
    return zscore
def detect_lsh_sweet(raw_text, lsh_model, lmbd, lsh_dim, tokenizer, llm_model, sweet_threshold=0.6):
    import math
    import torch
    from nltk.tokenize import sent_tokenize
    llm_model.eval()

    if isinstance(raw_text, list):
        raw_text = " ".join(raw_text)

    prompt = extract_prompt_from_text(raw_text, len_prompt=32)
    generated_text = raw_text[len(prompt):].strip()
    if not generated_text:
        return None

    gen_sents = sent_tokenize(generated_text)
    
    if len(gen_sents) > 1 and gen_sents[-1].strip()[-1] not in ".!?。！？\"'":
        gen_sents = gen_sents[:-1]

    if len(gen_sents) == 0:
        return None
    chunks = [prompt] + gen_sents
    total_green = 0
    total_tokens = 0
    vocab_size = len(tokenizer)
    last_end_pos = raw_text.find(chunks[0])
    if last_end_pos != -1:
        last_end_pos += len(chunks[0])
    else:
        last_end_pos = 0

    for i in range(1, len(chunks)):
        context_sentence = chunks[i-1]
        current_sentence = chunks[i]
        
        curr_start = raw_text.find(current_sentence, last_end_pos)
        if curr_start == -1:
            chunk_to_encode = current_sentence
            old_last_end_pos = last_end_pos
            chunk_tokens = tokenizer.encode(chunk_to_encode, add_special_tokens=False)
            full_text = raw_text[:last_end_pos] + " " + current_sentence
        else:
            curr_end = curr_start + len(current_sentence)
            chunk_to_encode = raw_text[last_end_pos : curr_end]
            old_last_end_pos = last_end_pos
            last_end_pos = curr_end
            
            full_text = raw_text[:curr_end] 
            context_text = raw_text[:old_last_end_pos]
            
            full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
            context_tokens = tokenizer.encode(context_text, add_special_tokens=False)
            chunk_tokens = full_tokens[len(context_tokens):]
            
        lsh_sig = lsh_model.get_hash([context_sentence])[0]
        from detection_utils import get_seed_from_lsh
        current_seed = get_seed_from_lsh(lsh_sig)
        rng = torch.Generator(device='cpu')
        rng.manual_seed(current_seed) 
        vocab_permutation = torch.randperm(vocab_size, generator=rng)
        greenlist = set(vocab_permutation[:int(vocab_size * lmbd)].tolist())

        num_chunk = len(chunk_tokens)
        if num_chunk == 0:
            continue
            
        inputs = tokenizer(full_text, return_tensors="pt").to(llm_model.device)
        with torch.no_grad():
            outputs = llm_model(**inputs)
            logits = outputs.logits[0]  
            
            if logits.shape[0] > num_chunk:
                chunk_logits = logits[-num_chunk - 1 : -1, :]
            else:
                chunk_logits = logits[:-1, :] 
            
        probs = torch.softmax(chunk_logits.float(), dim=-1)
        safe_probs = torch.clamp(probs, min=1e-9, max=1.0)
        chunk_entropies = (-torch.sum(probs * torch.log(safe_probs), dim=-1)).cpu().tolist()
        
        del inputs, outputs, logits, chunk_logits, probs
        torch.cuda.empty_cache()
        
        if len(chunk_entropies) < num_chunk:
            chunk_entropies = [10.0] * (num_chunk - len(chunk_entropies)) + chunk_entropies
            
        for idx, tok in enumerate(chunk_tokens):
            if chunk_entropies[idx] < sweet_threshold:
                continue
                
            if tok in greenlist:
                total_green += 1
            total_tokens += 1

    if total_tokens == 0:
        return None

    expected_green = lmbd * total_tokens
    variance = total_tokens * lmbd * (1 - lmbd)

    if variance == 0:
        return None

    zscore = (total_green - expected_green) / math.sqrt(variance)
    return zscore
def flatten_gens_and_paras(gens, paras):
    new_gens = []
    new_paras = []
    for gen, para in zip(gens, paras):
        min_len = min(len(gen), len(para))
        new_gens.extend(gen[:min_len])
        new_paras.extend(para[:min_len])
    return new_gens, new_paras

def get_roc_metrics(labels, preds):
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)

def get_roc_metrics_from_zscores(m, mp, h, dataset_path):
    mp = np.nan_to_num(mp)
    h = np.nan_to_num(h)
    len_z = len(mp)
    mp_fpr, mp_tpr, mp_area = get_roc_metrics(
        [1] * len_z + [0] * len_z, np.concatenate((mp, h[:len_z])))
    plt.plot(mp_fpr, mp_tpr)
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.title("ROC Curve")
    name = os.path.join(dataset_path, "roc_curve.png")
    plt.savefig(name)
    name = os.path.join(dataset_path, "fpr.npy")
    np.save(name, mp_fpr)
    name = os.path.join(dataset_path, "tpr.npy")
    np.save(name, mp_tpr)
    return mp_area, mp_fpr
def get_mask_from_seed_gpu(lsh_dim: int, accept_rate: float, seed: int):
    hash_key = 15485863
    n_bins = 2**lsh_dim
    n_accept = int(n_bins * accept_rate)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = torch.Generator(device=device) 
    rng.manual_seed(hash_key * seed)
    vocab_permutation = torch.randperm(n_bins, device=device, generator=rng)
    return vocab_permutation[:n_accept].tolist()

def extract_prompt_from_text(text, len_prompt=32):
    PUNCTS = '.,!?'
    tokens = text.split(' ')
    tokens = tokens[:len_prompt]
    new_text = ' '.join(tokens)
    prompts = []
    for p in PUNCTS:
        idx = new_text.find(p)
        if idx != -1:
            tokens = new_text[:idx + 1].split(" ")
            if len(tokens) > 3:
                prompts.append(new_text[:idx + 1])
    if len(prompts) == 0:
        prompts.append(new_text + ".")
    prompt = list(sorted(prompts, key=lambda x: len(x)))[0]
    return prompt

def detect_semstamp_official(raw_text, lsh_model, lmbd, lsh_dim, tokenizer=None):
    import math
    from nltk.tokenize import sent_tokenize
    
    if isinstance(raw_text, list):
        raw_text = " ".join(raw_text)
        
    prompt = extract_prompt_from_text(raw_text, len_prompt=32)
    text_ids = tokenizer.encode(raw_text, add_special_tokens=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    
    chunks = [prompt]
    current_num_sentences = len(sent_tokenize(prompt))
    last_chunk_end_idx = len(prompt_ids)
    
    for i in range(len(prompt_ids) + 1, len(text_ids) + 1):
        full_decoded_text = tokenizer.decode(text_ids[:i], skip_special_tokens=True)
        num_sents = len(sent_tokenize(full_decoded_text))
        
        if num_sents > current_num_sentences:
            new_text = tokenizer.decode(text_ids[last_chunk_end_idx:i], skip_special_tokens=True)
            if new_text.strip() == '':
                new_text = "."
            chunks.append(new_text)
            current_num_sentences = num_sents
            last_chunk_end_idx = i
            
    if last_chunk_end_idx < len(text_ids):
        new_text = tokenizer.decode(text_ids[last_chunk_end_idx:], skip_special_tokens=True)
        if new_text.strip() == '':
            new_text = "."
        chunks.append(new_text)
        
    if len(chunks) <= 1:
        return None
        
    total_green = 0
    total_sents = len(chunks) - 1
    lsh_seed = lsh_model.get_hash([chunks[0]])[0]
    from detection_utils import get_mask_from_seed_gpu
    accept_mask = get_mask_from_seed_gpu(lsh_dim, lmbd, lsh_seed)
    for i in range(1, len(chunks)):
        current_sentence = chunks[i]
        lsh_candidate = lsh_model.get_hash([current_sentence])[0]
        
        if lsh_candidate in accept_mask:
            total_green += 1
            
        lsh_seed = lsh_candidate
        accept_mask = get_mask_from_seed_gpu(lsh_dim, lmbd, lsh_seed)
        
    expected = lmbd * total_sents
    variance = total_sents * lmbd * (1 - lmbd)
    if variance == 0:
        return None
        
    zscore = (total_green - expected) / math.sqrt(variance)
    return zscore
def detect_kmeans(raw_text, embedder, lmbd, k_dim, cluster_centers, tokenizer):
    import math
    import torch
    from sampling_kmeans_utils import kmeans_predict
    from nltk.tokenize import sent_tokenize
    
    if isinstance(raw_text, list):
        raw_text = " ".join(raw_text)
        
    prompt = extract_prompt_from_text(raw_text, len_prompt=32)
    
    text_ids = tokenizer.encode(raw_text, add_special_tokens=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    
    chunks = [prompt] 
    current_num_sentences = len(sent_tokenize(prompt))
    last_chunk_end_idx = len(prompt_ids)
    
    for i in range(len(prompt_ids) + 1, len(text_ids) + 1):
        full_decoded_text = tokenizer.decode(text_ids[:i], skip_special_tokens=True)
        num_sents = len(sent_tokenize(full_decoded_text))
        
        if num_sents > current_num_sentences:
            new_text = tokenizer.decode(text_ids[last_chunk_end_idx:i], skip_special_tokens=True)
            if new_text.strip() == '':
                new_text = "."
            chunks.append(new_text)
            current_num_sentences = num_sents
            last_chunk_end_idx = i
            
    if last_chunk_end_idx < len(text_ids):
        new_text = tokenizer.decode(text_ids[last_chunk_end_idx:], skip_special_tokens=True)
        if new_text.strip() == '':
            new_text = "."
        chunks.append(new_text)
        
    if len(chunks) <= 1:
        return None
    hash_key = 15485863
    total_green = 0
    total_sents = len(chunks) - 1
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    embeddings = embedder.encode(chunks, convert_to_tensor=True)
    cluster_ids = kmeans_predict(embeddings, cluster_centers=cluster_centers, distance='cosine', device=device)
    
    for i in range(1, len(chunks)):
        prev_id = cluster_ids[i-1]
        curr_id = cluster_ids[i]
        
        rng = torch.Generator(device=device)
        rng.manual_seed(prev_id.item() * hash_key)
        num_accept = int(k_dim * lmbd)
        mask = torch.randperm(k_dim, device=device, generator=rng)[:num_accept]
        
        if curr_id in mask:
            total_green += 1
            
    expected = lmbd * total_sents
    variance = total_sents * lmbd * (1 - lmbd)
    if variance == 0:
        return None
        
    zscore = (total_green - expected) / math.sqrt(variance)
    return zscore

def detect_kgw_sweet(raw_text, fraction, tokenizer, llm_model, sweet_threshold=0.6, hash_key=15485863):
    import math
    import torch
    from detection_utils import extract_prompt_from_text

    llm_model.eval()

    if isinstance(raw_text, list):
        raw_text = " ".join(raw_text)

    prompt = extract_prompt_from_text(raw_text, len_prompt=32)
    
    text_ids = tokenizer.encode(raw_text, add_special_tokens=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    
    gen_start_idx = len(prompt_ids)
    if gen_start_idx >= len(text_ids):
        return None 

    inputs = tokenizer(raw_text, return_tensors="pt").to(llm_model.device)
    with torch.no_grad():
        outputs = llm_model(**inputs)
        logits = outputs.logits[0]

    probs = torch.softmax(logits.float(), dim=-1)
    safe_probs = torch.clamp(probs, min=1e-9, max=1.0)
    entropies = (-torch.sum(probs * torch.log(safe_probs), dim=-1)).cpu().tolist()

    del inputs, outputs, logits, probs, safe_probs
    torch.cuda.empty_cache()

    total_green = 0
    total_tokens = 0
    vocab_size = len(tokenizer)

    for k in range(gen_start_idx, len(text_ids)):
        prev_token = text_ids[k-1]
        curr_token = text_ids[k]
        
        # logits[k-1] 预测的是 text_ids[k]
        entropy = entropies[k-1] 
        
        if entropy < sweet_threshold:
            continue
        seed = (hash_key * prev_token) % (2**31 - 1)
        rng = torch.Generator(device=llm_model.device)
        rng.manual_seed(seed)
        
        greenlist_size = int(vocab_size * fraction)
        vocab_permutation = torch.randperm(vocab_size, generator=rng, device=llm_model.device)
        greenlist_ids = set(vocab_permutation[:greenlist_size].cpu().tolist())
        
        if curr_token in greenlist_ids:
            total_green += 1
        total_tokens += 1

    if total_tokens == 0:
        return None

    # 5. 计算 Z-score 显著性
    expected_green = fraction * total_tokens
    variance = total_tokens * fraction * (1 - fraction)

    if variance == 0:
        return None

    zscore = (total_green - expected_green) / math.sqrt(variance)
    return zscore
