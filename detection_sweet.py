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
from detection_utils import detect_lsh_sweet, flatten_gens_and_paras 
from transformers import AutoTokenizer, AutoModelForCausalLM
from detection_utils import detect_kgw_sweet
import torch
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='hf dataset containing text and para_text columns')
    parser.add_argument('--human_text', help='hf dataset containing text column', default="data/c4-human")
    parser.add_argument('--detection_mode', choices=['kmeans', 'lsh', 'kgw'],
                    help='detection mode. lsh for semstamp, kmeans for k-semstamp, kgw for official SWEET')
    parser.add_argument('--cc_path', type=str, help='path to cluster centers')
    parser.add_argument('--embedder', type=str, help='sentence embedder')
    parser.add_argument('--model', type=str, help='backbone LM for text generation', default='facebook/opt-1.3b')
    parser.add_argument('--sp_dim', type=int, default=3,
                        help='dimension of the subspaces. default 3 for sstamp and 8 for ksstamp')
    parser.add_argument('--max_new_tokens', type=int, default=205)
    parser.add_argument('--lmbd', type=float, default=0.25, help='ratio of valid sentences')
    parser.add_argument('--sweet_threshold', type=float, default=0.6, help='Entropy threshold for SWEET')

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
    print("🧠 正在加载大语言模型用于 SWEET 熵值计算 (这可能需要一点时间)...")
    llm_model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.float16)
    
    # ksemstamp detection
    if args.detection_mode == 'kmeans':
        cluster_centers = torch.load(args.cc_path)
        embedder = SentenceTransformer(args.embedder)
        for i in trange(0, len(gens), 1, desc='kmeans_detection'):
            gen_sents = gens[i] if type(gens[i]) == list else sent_tokenize(gens[i])
            z_score = detect_kmeans(sents=gen_sents, embedder=embedder, lmbd=args.lmbd,
                                    k_dim=args.sp_dim, cluster_centers=cluster_centers)
            z_scores.append(z_score)

        for i in trange(0, len(human_texts), 1, desc='kmeans_human'):
            sents = sent_tokenize(human_texts[i])
            z_score = detect_kmeans(sents=sents, embedder=embedder, lmbd=args.lmbd,
                                    k_dim=args.sp_dim, cluster_centers=cluster_centers)
            human_scores.append(z_score)
            
    # semstamp detection
    elif args.detection_mode == 'lsh':
        lsh_model_class = SBERTLSHModel
        lsh_model = lsh_model_class(
            lsh_model_path=args.embedder, device='cuda', batch_size=1, lsh_dim=args.sp_dim, sbert_type='base'
        )
        for i in trange(0, len(gens), 1):
            text_sents = gens[i] if type(gens[i]) == list else sent_tokenize(gens[i])
            z_score = detect_lsh_sweet(
                raw_text=gens[i], lsh_model=lsh_model, 
                lmbd=args.lmbd, lsh_dim=args.sp_dim, tokenizer=tokenizer, llm_model=llm_model, 
                sweet_threshold=args.sweet_threshold 
            )
            # 不是 None 才加进去
            if z_score is not None:
                z_scores.append(z_score)
            
            # --- 洗稿部分的检测 ---
            if paras != None:
                para_sents = paras[i] if type(paras[i]) == list else sent_tokenize(paras[i])
                para_z_score = detect_lsh_sweet(
                raw_text=paras[i], lsh_model=lsh_model, 
                lmbd=args.lmbd, lsh_dim=args.sp_dim, tokenizer=tokenizer, llm_model=llm_model, 
                sweet_threshold=args.sweet_threshold 
            )
                
                if para_z_score is not None:
                    para_scores.append(para_z_score)

        for i in trange(0, len(human_texts), 1):
            # 👇 1. 先安全地把数据转成长字符串（兼容 List 格式）
            raw_human = human_texts[i]
            raw_human_str = " ".join(raw_human) if isinstance(raw_human, list) else str(raw_human)
            
            # 👇 2. 截取前 500 个单词
            truncated_human_text = " ".join(raw_human_str.split()[:500]) 
            
            # 👇 3. 进行分句和检测
            sents = sent_tokenize(truncated_human_text)
            z_score = detect_lsh_sweet(
                raw_text=truncated_human_text, lsh_model=lsh_model, 
                lmbd=args.lmbd, lsh_dim=args.sp_dim, tokenizer=tokenizer, llm_model=llm_model, 
                sweet_threshold=args.sweet_threshold 
            )
            # 不是 None 才加进去
            if z_score is not None:
                human_scores.append(z_score)
    elif args.detection_mode == 'kgw':
        for i in trange(0, len(gens), 1, desc="KGW SWEET Gen"):
            z_score = detect_kgw_sweet(
                raw_text=gens[i], fraction=args.lmbd, 
                tokenizer=tokenizer, llm_model=llm_model, 
                sweet_threshold=args.sweet_threshold
            )
            if z_score is not None:
                z_scores.append(z_score)

            # 洗稿攻击检测
            if paras != None:
                para_z_score = detect_kgw_sweet(
                    raw_text=paras[i], fraction=args.lmbd, 
                    tokenizer=tokenizer, llm_model=llm_model, 
                    sweet_threshold=args.sweet_threshold
                )
                if para_z_score is not None:
                    para_scores.append(para_z_score)

        # 人类文本检测 (计算 FPR 基线)
        for i in trange(0, len(human_texts), 1, desc="KGW SWEET Human"):
            raw_human = human_texts[i]
            raw_human_str = " ".join(raw_human) if isinstance(raw_human, list) else str(raw_human)
            truncated_human_text = " ".join(raw_human_str.split()[:500]) 

            z_score = detect_kgw_sweet(
                raw_text=truncated_human_text, fraction=args.lmbd, 
                tokenizer=tokenizer, llm_model=llm_model, 
                sweet_threshold=args.sweet_threshold
            )
            if z_score is not None:
                human_scores.append(z_score)
    z_score_name = os.path.join(args.dataset_path, "z_scores.npy")
    
    para_score_name = os.path.join(args.dataset_path, "para_z_scores.npy") 
    human_score_name = os.path.join(args.dataset_path, "human_z_scores.npy")

    np.save(z_score_name, z_scores)
    np.save(para_score_name, para_scores) 
    np.save(human_score_name, human_scores)
    print("\n" + "="*80)
    print(" 开始排查低分（False Negatives）离群点...")

    # 1. 计算 FPR=1% 时的具体 Z-score 阈值
    threshold_1_percent = np.percentile(human_scores, 99)
    print(f"当前 TPR@1% 对应的 Z-score 及格线约为: {threshold_1_percent:.4f}")

    # 2. 找出所有未过线的生成文本的索引
    failed_indices = [i for i, z in enumerate(z_scores) if z < threshold_1_percent]

    print(f"共有 {len(failed_indices)}/{len(z_scores)} 个生成样本未能通过 1% 阈值检测。")
    print("正在打印 Z-score 垫底的前 20 个样本进行人工诊断：")

    # 3. 按分数从低到高排序，打印最差的那些
    failed_indices_sorted = sorted(failed_indices, key=lambda i: z_scores[i])

    for rank, idx in enumerate(failed_indices_sorted[:20]):
        raw_text_data = gens[idx]
        z_val = z_scores[idx]
        
        if isinstance(raw_text_data, list):
            sents = raw_text_data
            text_str = " ".join(raw_text_data)
        else:
            sents = sent_tokenize(raw_text_data)
            text_str = raw_text_data
            
        print("-" * 60)
        print(f"🔻 排名: 倒数第 {rank + 1} | 原始索引: {idx}")
        print(f"🔻 Z-score: {z_val:.4f} (距及格线差 {threshold_1_percent - z_val:.4f})")
        print(f"🔻 统计: 包含 {len(sents)} 个句子, 总字符数: {len(text_str)}")
        print(f"🔻 文本内容:\n{text_str}")
    print("="*80 + "\n")
    results_path = os.path.join(args.dataset_path, "results_sweet.csv")
    from sklearn.metrics import roc_curve, roc_auc_score
    print("Evaluating SWEET metrics...")
    
    # --- 1. 计算未受攻击（纯文本 gens）的指标 ---
    labels_pure = [1] * len(z_scores) + [0] * len(human_scores) 
    scores_pure = list(z_scores) + list(human_scores)
    pure_auroc = roc_auc_score(labels_pure, scores_pure)
    
    # 计算未受攻击的 TPR@1% 和 TPR@5% 
    fpr_pure, tpr_pure, _ = roc_curve(labels_pure, scores_pure)
    pure_tpr1 = tpr_pure[fpr_pure <= 0.01][-1] if len(tpr_pure[fpr_pure <= 0.01]) > 0 else 0.0
    pure_tpr5 = tpr_pure[fpr_pure <= 0.05][-1] if len(tpr_pure[fpr_pure <= 0.05]) > 0 else 0.0
    
    # --- 2. 计算受攻击（洗稿 paras）的指标 ---
    if len(para_scores) > 0:
        labels_attack = [1] * len(para_scores) + [0] * len(human_scores) 
        scores_attack = list(para_scores) + list(human_scores)
        attack_auroc = roc_auc_score(labels_attack, scores_attack)
        
        # 计算受攻击的 TPR@1% 和 TPR@5%
        fpr, tpr, thresholds = roc_curve(labels_attack, scores_attack)
        attack_tpr1 = tpr[fpr <= 0.01][-1] if len(tpr[fpr <= 0.01]) > 0 else 0.0
        attack_tpr5 = tpr[fpr <= 0.05][-1] if len(tpr[fpr <= 0.05]) > 0 else 0.0
    else:
        attack_auroc = attack_tpr1 = attack_tpr5 = 0.0
        print("⚠️ 提示：当前未检测到洗稿数据(para_text)，受攻击指标暂记为 0。")
    
    # --- 3. 打印两个结果 ---
    print("="*80)
    print(f"✅【SWEET 未受攻击 (原文本) 指标】 AUC: {pure_auroc:.3f} | TPR@1%: {pure_tpr1:.3f} | TPR@5%: {pure_tpr5:.3f}")
    if len(para_scores) > 0:
        print(f"🚨【SWEET 重述攻击后指标】   AUC: {attack_auroc:.3f} | TPR@1%: {attack_tpr1:.3f} | TPR@5%: {attack_tpr5:.3f}")
    print("="*80)
    auroc, fpr1, fpr5 = attack_auroc, attack_tpr1, attack_tpr5
    bert_score = 0.0 
    
    metrics = [f"{pure_auroc:.3f}", f"{pure_tpr1:.3f}", f"{pure_tpr5:.3f}", f"{bert_score:.3f}"]
    columns = ["auroc", "fpr1", "fpr5", "bert_score"]
    df = pd.DataFrame(data=[metrics], columns=columns)
    df.to_csv(results_path, sep="\t", index=False)