import torch
import numpy as np
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

def train_kmeans_centers(dataset_path, embedder_path, k_dim=16, sample_size=20000):
    print("1. 加载训练数据...")
    # 使用你的训练集提取语义分布
    dataset = load_from_disk(dataset_path)
    texts = dataset['text'][:sample_size]
    
    # 将列表展开成单独的句子
    sentences = []
    for t in texts:
        if isinstance(t, list):
            sentences.extend(t)
        else:
            sentences.append(t)
            
    print(f"2. 加载精调模型并编码 {len(sentences)} 个句子 (这需要一小会儿)...")
    embedder = SentenceTransformer(embedder_path, device='cuda')
    embeddings = embedder.encode(sentences, batch_size=256, show_progress_bar=True)
    
    print(f"3. 运行 KMeans 聚类 (K={k_dim})...")
    # random_state 保证每次聚类中心完全一致，可复现
    kmeans = KMeans(n_clusters=k_dim, random_state=42, n_init="auto")
    kmeans.fit(embeddings)
    
    print("4. 保存聚类中心...")
    centers = torch.tensor(kmeans.cluster_centers_)
    # 保存为一个 .pt 文件供 sampling.py 和 detection.py 读取
    torch.save(centers, 'cluster_centers.pt')
    print(f"✅ 成功保存 {k_dim} 个聚类中心到 cluster_centers.pt！")

if __name__ == '__main__':
    train_kmeans_centers(
        dataset_path="dataset/c4/80train20test-8000-00:08000-para4/train", # 使用训练集
        embedder_path="/home/haojifei/dev_projects/nlp_projects/Learn-LSH/my_watermark_embedder_finetuned",
        k_dim=16 # 推荐聚成 16 类，这是一个极佳的平衡点
    )