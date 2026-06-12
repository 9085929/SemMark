from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer, models
import torch
import os
import shutil

base_model_path = "/home/haojifei/dev_resource/huggingface/models/Qwen/Qwen3-Embedding-8B"
lora_weights_path = "qwen_lsh8B_lora_rewnews" # 训练脚本里保存 LoRA 的目录
save_path = "/home/haojifei/dev_projects/nlp_projects/Learn-LSH/my_watermark_embedder8B_finetuned_booksum"
temp_hf_path = "./temp_merged_hf" 

print("1. 加载原生 HF 躯壳并注入 LoRA 权重...")
base_model = AutoModel.from_pretrained(base_model_path, trust_remote_code=True, device_map="cpu")
model_with_lora = PeftModel.from_pretrained(base_model, lora_weights_path)

print("2. 熔铸合并为完整模型 (Merge and Unload)...")
merged_model = model_with_lora.merge_and_unload()

print("3. 保存临时 HF 格式...")
merged_model.save_pretrained(temp_hf_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.save_pretrained(temp_hf_path)

print("4. 正在封装为 SentenceTransformer 标准格式...")
# 读取刚才合并好的临时 HF 模型
word_embedding_model = models.Transformer(
    temp_hf_path,
    model_args={"trust_remote_code": True},
    tokenizer_args={"trust_remote_code": True}
)

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
norm_model = models.Normalize() 
model = SentenceTransformer(modules=[word_embedding_model, pooling_model, norm_model])

print("5. 导出最终模型...")
model.save(save_path)

if os.path.exists(temp_hf_path):
    shutil.rmtree(temp_hf_path)

print(f"🎉 搞定！合并并封装完成，已保存至: {save_path}")
print("   现在你可以直接运行 sampling.py 和 detection_sweet.py 了！")
