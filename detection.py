import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from sbert_lsh_model import SBERTLSHModel
from tqdm import trange
from sentence_transformers import SentenceTransformer
import argparse
from datasets import load_from_disk
from nltk.tokenize import sent_tokenize
import os
import torch
import numpy as np
#from detection_utils import detect_lsh, flatten_gens_and_paras, detect_semstamp_official
from detection_utils import detect_lsh, flatten_gens_and_paras, detect_semstamp_official, detect_kmeans
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='hf dataset containing text and para_text columns')
    parser.add_argument('--human_text', help='hf dataset containing text column', default="data/c4-human")
    parser.add_argument('--detection_mode', choices=['kmeans', 'lsh', 'semstamp'],
                        help='detection mode. lsh for semstamp and kmeans for k-semstamp')
    parser.add_argument('--cc_path', type=str, help='path to cluster centers')
    parser.add_argument('--embedder', type=str, help='sentence embedder')
    parser.add_argument('--model', type=str, help='backbone LM for text generation', default='facebook/opt-1.3b')
    parser.add_argument('--sp_dim', type=int, default=3,
                        help='dimension of the subspaces. default 3 for sstamp and 8 for ksstamp')
    parser.add_argument('--max_new_tokens', type=int, default=205)
    parser.add_argument('--lmbd', type=float, default=0.25, help='ratio of valid sentences')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    full_ds = load_from_disk(args.dataset_path)
    gens = []
    paras = []
    
    if type(full_ds).__name__ == 'DatasetDict' or isinstance(full_ds, dict):
        for split in full_ds.keys():
            gens.extend(full_ds[split]['text'])
            if 'para_text1' in full_ds[split].column_names:
                paras.extend(full_ds[split]['para_text1'])
    else:
        gens.extend(full_ds['text'])
        if 'para_text1' in full_ds.column_names:
            paras.extend(full_ds['para_text1'])
            
    if len(paras) == 0:
        paras = None
    human_texts = load_from_disk(args.human_text)['text']
    z_scores, para_scores, human_scores = [], [], []
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # ksemstamp detection
    if args.detection_mode == 'kmeans':
        cluster_centers = torch.load(args.cc_path)
        embedder = SentenceTransformer(args.embedder)
        
        for i in trange(0, len(gens), 1, desc='kmeans_detection'):
            # 🌟 这里多加一个 tokenizer=tokenizer
            z_score = detect_kmeans(raw_text=gens[i], embedder=embedder, lmbd=args.lmbd,
                                    k_dim=args.sp_dim, cluster_centers=cluster_centers, tokenizer=tokenizer)
            if z_score is not None:
                z_scores.append(z_score)
            
            if paras != None:
                # 🌟 这里多加一个 tokenizer=tokenizer
                para_z_score = detect_kmeans(raw_text=paras[i], embedder=embedder, lmbd=args.lmbd,
                                        k_dim=args.sp_dim, cluster_centers=cluster_centers, tokenizer=tokenizer)
                if para_z_score is not None:
                    para_scores.append(para_z_score)

        for i in trange(0, len(human_texts), 1, desc='kmeans_human'):
            # 🌟 这里多加一个 tokenizer=tokenizer
            z_score = detect_kmeans(raw_text=human_texts[i], embedder=embedder, lmbd=args.lmbd,
                                    k_dim=args.sp_dim, cluster_centers=cluster_centers, tokenizer=tokenizer)
            if z_score is not None:
                human_scores.append(z_score)
            
    # semstamp detection
    elif args.detection_mode == 'lsh':
        lsh_model_class = SBERTLSHModel
        lsh_model = lsh_model_class(
            lsh_model_path=args.embedder, device='cuda', batch_size=1, lsh_dim=args.sp_dim, sbert_type='base'
        )
        for i in trange(0, len(gens), 1):
            text_sents = gens[i] if type(gens[i]) == list else sent_tokenize(gens[i])
            z_score = detect_lsh(
                raw_text=gens[i], lsh_model=lsh_model,  
                lmbd=args.lmbd, lsh_dim=args.sp_dim, tokenizer=tokenizer
            )
            # 不是 None 才加进去
            if z_score is not None:
                z_scores.append(z_score)
            
            # --- 洗稿部分的检测 ---
            if paras != None:
                para_sents = paras[i] if type(paras[i]) == list else sent_tokenize(paras[i])
                para_z_score = detect_lsh(
                raw_text=paras[i], lsh_model=lsh_model, # 直接传 paras[i]
                lmbd=args.lmbd, lsh_dim=args.sp_dim, tokenizer=tokenizer
            )
                # 同样加上安全气囊，防止脏数据导致报错
                if para_z_score is not None:
                    para_scores.append(para_z_score)

        for i in trange(0, len(human_texts), 1):
            sents = human_texts[i] if type(human_texts[i]) == list else sent_tokenize(human_texts[i])
            z_score = detect_lsh(
                raw_text=human_texts[i], lsh_model=lsh_model, # 直接传 human_texts[i]
                lmbd=args.lmbd, lsh_dim=args.sp_dim, tokenizer=tokenizer
            )
            # 不是 None 才加进去
            if z_score is not None:
                human_scores.append(z_score)
    #  SemStamp 官方检测通道
    elif args.detection_mode == 'semstamp':
        lsh_model_class = SBERTLSHModel
        lsh_model = lsh_model_class(
            lsh_model_path=args.embedder, device='cuda', batch_size=1, lsh_dim=args.sp_dim, sbert_type='base'
        )
        for i in trange(0, len(gens), 1):
            z_score = detect_semstamp_official(gens[i], lsh_model, args.lmbd, args.sp_dim, tokenizer=tokenizer)
            if z_score is not None:
                z_scores.append(z_score)
                
            if paras != None:
                para_z_score = detect_semstamp_official(paras[i], lsh_model, args.lmbd, args.sp_dim, tokenizer=tokenizer)
                if para_z_score is not None:
                    para_scores.append(para_z_score)

        for i in trange(0, len(human_texts), 1):
            z_score = detect_semstamp_official(human_texts[i], lsh_model, args.lmbd, args.sp_dim, tokenizer=tokenizer)
            if z_score is not None:
                human_scores.append(z_score)

    z_score_name = os.path.join(args.dataset_path, "z_scores.npy")
    # 解除保存封印
    para_score_name = os.path.join(args.dataset_path, "para_z_scores.npy") 
    human_score_name = os.path.join(args.dataset_path, "human_z_scores.npy")

    np.save(z_score_name, z_scores)
    np.save(para_score_name, para_scores) # 解除保存封印
    np.save(human_score_name, human_scores)

    results_path = os.path.join(args.dataset_path, "results.csv")
    from sklearn.metrics import roc_curve, roc_auc_score
    print("Evaluating metrics (Under Paraphrase Attack)...")
    from sklearn.metrics import roc_curve, roc_auc_score
    print("Evaluating metrics (Standard LSH)...")
    
    # --- 1. 计算未受攻击（纯文本 gens）的指标 ---
    labels_pure = [1] * len(z_scores) + [0] * len(human_scores) 
    scores_pure = list(z_scores) + list(human_scores)
    pure_auroc = roc_auc_score(labels_pure, scores_pure)
    
    fpr_pure, tpr_pure, _ = roc_curve(labels_pure, scores_pure)
    pure_tpr1 = tpr_pure[fpr_pure <= 0.01][-1] if len(tpr_pure[fpr_pure <= 0.01]) > 0 else 0.0
    pure_tpr5 = tpr_pure[fpr_pure <= 0.05][-1] if len(tpr_pure[fpr_pure <= 0.05]) > 0 else 0.0
    
    # --- 2. 计算受攻击（洗稿 paras）的指标 ---
    if len(para_scores) > 0:
        labels_attack = [1] * len(para_scores) + [0] * len(human_scores) 
        scores_attack = list(para_scores) + list(human_scores)
        attack_auroc = roc_auc_score(labels_attack, scores_attack)
        
        fpr, tpr, thresholds = roc_curve(labels_attack, scores_attack)
        attack_tpr1 = tpr[fpr <= 0.01][-1] if len(tpr[fpr <= 0.01]) > 0 else 0.0
        attack_tpr5 = tpr[fpr <= 0.05][-1] if len(tpr[fpr <= 0.05]) > 0 else 0.0
    else:
        attack_auroc = attack_tpr1 = attack_tpr5 = 0.0
        print("⚠️ 提示：当前未检测到洗稿数据(para_text)，受攻击指标暂记为 0。")
    
    # --- 3. 打印两个结果 ---
    print("="*80)
    print(f"✅【标准 LSH 未受攻击指标】 AUC: {pure_auroc:.3f} | TPR@1%: {pure_tpr1:.3f} | TPR@5%: {pure_tpr5:.3f}")
    if len(para_scores) > 0:
        print(f"🚨【标准 LSH 重述攻击后指标】 AUC: {attack_auroc:.3f} | TPR@1%: {attack_tpr1:.3f} | TPR@5%: {attack_tpr5:.3f}")
    print("="*80)
    
    metrics = [f"{pure_auroc:.3f}", f"{pure_tpr1:.3f}", f"{pure_tpr5:.3f}", "0.000"]
    columns = ["auroc", "fpr1", "fpr5", "bert_score"]
    df = pd.DataFrame(data=[metrics], columns=columns)
    df.to_csv(results_path, sep="\t", index=False)