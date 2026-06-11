# uncomment to install rjieba which is needed for the tokenizer
# !pip install rjieba
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained(
    "junnyu/roformer_chinese_base", dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("junnyu/roformer_chinese_base")

input_ids = tokenizer("水在零度时会[MASK]", return_tensors="pt").to(model.device)
outputs = model(**input_ids)
decoded = tokenizer.batch_decode(outputs.logits.argmax(-1), skip_special_tokens=True)
print(decoded)
