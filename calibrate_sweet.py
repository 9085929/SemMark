import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

DATA_PATH = "dataset/benchmark_datasets/realnews_10k-8000-pegasus-merged" # 原始数据集
MODEL_PATH = "/home/haojifei/dev_resource/huggingface/models/Qwen/Qwen3-8B"
SKIP_LINES = 8000       # 跳过前 8000 条已使用的数据
NUM_SAMPLES = 2000       # 跑 2000 条验证集数据
GAMMA = 0.5             # 绿名单比例 
DELTA = 2.0             # 加水印时的偏置参数 
MC_ITER = 30            # 蒙特卡洛模拟次数 
MAX_LENGTH = 512        # 单条文本最大长度

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("🧠 正在加载 Qwen3-8B 模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    from datasets import load_from_disk 
    print(f"📖 正在从 HuggingFace 磁盘缓存读取数据: {DATA_PATH} ...")
    texts = []
    
    # 加载 DatasetDict
    full_dataset = load_from_disk(DATA_PATH)
    
    split_to_use = 'train' if 'train' in full_dataset else list(full_dataset.keys())[0]
    ds = full_dataset[split_to_use]
    
    print(f"📂 使用的划分: {split_to_use}, 总数据量: {len(ds)}")
    
    for i, item in enumerate(ds):
        if i < SKIP_LINES:
            continue
        if len(texts) >= NUM_SAMPLES:
            break
            
        try:
            raw_text = " ".join(item['text']) if isinstance(item['text'], list) else item['text']
            
            if raw_text.strip():
                texts.append(raw_text)
        except Exception as e:
            continue

    all_H = []
    all_PG = []

    print("🚀 开始进行前向传播与蒙特卡洛模拟分析...")
    for text in tqdm(texts, desc="Analyzing Logits"):
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
        input_ids = tokens.input_ids
        
        if input_ids.shape[1] < 2:
            continue
            
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, :-1, :] 
            seq_len, vocab_size = logits.shape
            
            # 1. 计算原始香农熵 H
            probs_orig = torch.softmax(logits.float(), dim=-1)
            safe_probs = torch.clamp(probs_orig, min=1e-9, max=1.0)
            H = -torch.sum(probs_orig * torch.log(safe_probs), dim=-1)
            
            # 2. 蒙特卡洛模拟：计算加水印后的绿名单预期命中率 P_G
            # 通过多次随机切分绿名单，模拟大样本下的期望值
            PG_sum = torch.zeros(seq_len, device=device)
            for _ in range(MC_ITER):
                # 随机生成绿名单掩码 (1表示在绿名单，0表示在红名单)
                green_mask = (torch.rand(seq_len, vocab_size, device=device) < GAMMA).float()
                
                # 对绿名单里的词加上偏置 DELTA
                watermarked_logits = logits.float() + DELTA * green_mask
                watermarked_probs = torch.softmax(watermarked_logits, dim=-1)
                
                # 累加本次模拟中，选出绿名单词的概率
                PG_sum += torch.sum(watermarked_probs * green_mask, dim=-1)
                
            # 求期望 P_G
            P_G = PG_sum / MC_ITER
            
            # 存入全局列表
            all_H.extend(H.cpu().tolist())
            all_PG.extend(P_G.cpu().tolist())
            
        # 强制清理显存
        del tokens, outputs, logits, probs_orig, safe_probs, PG_sum, green_mask, watermarked_logits, watermarked_probs
        torch.cuda.empty_cache()

    print(f"\n✅ 数据提取完毕！共收集了 {len(all_H)} 个有效 Token 的熵值和概率预期。")
    
    # 转换为 Numpy 数组进行高速向量化运算
    H_arr = np.array(all_H)
    PG_arr = np.array(all_PG)
    
    # 准备遍历的阈值 tau
    tau_values = np.arange(0.0, 10.0, 0.1)
    Z_expected_list = []
    
    print("📈 正在寻找理论最优熵阈值...")
    best_tau = 0.0
    max_Z = -1.0
    
    for tau in tau_values:
        valid_mask = H_arr > tau
        N_h = np.sum(valid_mask)
        
        if N_h == 0:
            Z_expected_list.append(0.0)          
            
        # 根据 SWEET 的理论公式计算预期 Z 分数
        E_G = np.sum(PG_arr[valid_mask])
        variance = N_h * GAMMA * (1 - GAMMA)
        
        if variance == 0:
            Z_expected = 0.0
        else:
            Z_expected = (E_G - GAMMA * N_h) / np.sqrt(variance)
            
        Z_expected_list.append(Z_expected)
        
        if Z_expected > max_Z:
            max_Z = Z_expected
            best_tau = tau

    print("="*60)
    print(f"🏆 【寻找完成】理论最优熵阈值 (Tau): {best_tau:.1f}")
    print(f"🎯 在该阈值下的理论最大 Z-score: {max_Z:.3f}")
    print("="*60)
    
    # 绘制超参寻优曲线并保存
    plt.figure(figsize=(10, 6))
    plt.plot(tau_values, Z_expected_list, label="Expected Z-score", color="blue", linewidth=2)
    plt.axvline(x=best_tau, color='red', linestyle='--', label=f"Optimal $\\tau$ = {best_tau:.1f}")
    plt.title("SWEET Entropy Threshold Calibration (C4 Natural Language)", fontsize=14)
    plt.xlabel("Entropy Threshold ($\\tau$)", fontsize=12)
    plt.ylabel("Expected Z-score ($Z_{expected}$)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    save_fig_path = "sweet_threshold_calibration1.png"
    plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    print(f"📊 完美！校准曲线已保存为: {save_fig_path}")

if __name__ == "__main__":
    main()
