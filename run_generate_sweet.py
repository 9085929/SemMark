import argparse
from tqdm import tqdm
import torch
import os
import time  
from nltk.tokenize import sent_tokenize
from datasets import load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, LogitsProcessor

class KGW_SWEET_LogitsWarper(LogitsProcessor):
    def __init__(self, vocab_size, fraction=0.5, strength=2.0, hash_key=15485863, entropy_threshold=0.6):
        self.vocab_size = vocab_size
        self.fraction = fraction       # 绿名单比例 
        self.strength = strength       # 水印强度
        self.hash_key = hash_key       # 哈希密钥
        self.entropy_threshold = entropy_threshold 

    def __call__(self, input_ids, scores):
        probs = torch.softmax(scores, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1) 
        
        watermarked_scores = scores.clone()
        
        for b_idx in range(input_ids.shape[0]):
            
            if entropy[b_idx] < self.entropy_threshold:
                continue
                
            if input_ids.shape[1] > 0:
                prev_token = input_ids[b_idx, -1].item()
            else:
                prev_token = 0
                
            seed = (self.hash_key * prev_token) % (2**31 - 1)
            
            rng = torch.Generator(device=scores.device)
            rng.manual_seed(seed)
            
            greenlist_size = int(self.vocab_size * self.fraction)
            vocab_permutation = torch.randperm(self.vocab_size, generator=rng, device=scores.device)
            greenlist_ids = vocab_permutation[:greenlist_size]
            
            watermarked_scores[b_idx, greenlist_ids] += self.strength
            
        return watermarked_scores


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
    return list(sorted(prompts, key=lambda x: len(x)))[0]


def main(args):
    print("Loading Model and Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        device_map='auto', 
        torch_dtype=torch.float16, 
        trust_remote_code=True
    )
    model.eval()

    kgw_sweet_warper = KGW_SWEET_LogitsWarper(
        vocab_size=len(tokenizer),
        fraction=args.fraction,
        strength=args.strength,
        hash_key=args.wm_key,
        entropy_threshold=args.sweet_threshold
    )

    watermark_processor = LogitsProcessorList([kgw_sweet_warper])

    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_from_disk(args.dataset_path)
    
    if args.num_test is not None and args.num_test < len(dataset):
        dataset = dataset.select(range(args.num_test))

    new_data = []

    print("Starting generation...")
    for item in tqdm(dataset, total=len(dataset)):
        raw_text = " ".join(item['text']) if isinstance(item['text'], list) else item['text']
        prefix = extract_prompt_from_text(raw_text, 32)

        batch = tokenizer(prefix, truncation=True, return_tensors="pt").to(model.device)
        num_tokens = batch['input_ids'].size(1)
        torch.cuda.synchronize(model.device)
        start_time = time.perf_counter()
        with torch.inference_mode():
            generation = model.generate(
                **batch,
                logits_processor=watermark_processor,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                do_sample=True,
                temperature=0.7, 
                top_p=args.top_p,
            )
            gen_text = tokenizer.batch_decode(generation[:, num_tokens:], skip_special_tokens=True)[0]
        torch.cuda.synchronize(model.device)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        gen_text_clean = gen_text.strip()
        generated_sents = sent_tokenize(gen_text_clean)
        num_sents = len(generated_sents)
        actual_num_tokens = len(tokenizer.encode(gen_text_clean, add_special_tokens=False))
        
        item['elapsed_time'] = elapsed_time
        item['num_sents'] = num_sents
        item['num_tokens'] = actual_num_tokens
        item['sec_per_sent'] = elapsed_time / num_sents if num_sents > 0 else 0
        item['tokens_per_sec'] = actual_num_tokens / elapsed_time if elapsed_time > 0 else 0

        item['text'] = prefix + " " + gen_text
        item['is_watermarked'] = True
        new_data.append(item)
    output_dataset = Dataset.from_list(new_data)
    os.makedirs(args.output_dir, exist_ok=True)
    output_dataset.save_to_disk(args.output_dir)
    print(f"✅ Finished! Official KGW-SWEET Baseline saved to {args.output_dir}")
    avg_sec_per_sent = sum(output_dataset['sec_per_sent']) / len(output_dataset)
    avg_tps = sum(output_dataset['tokens_per_sec']) / len(output_dataset)
    print(f"\n📊 评测完成！平均生成速度: {avg_sec_per_sent:.3f} 秒/句, 吞吐量: {avg_tps:.1f} Tokens/s")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--wm_key", type=int, default=15485863)
    parser.add_argument("--max_new_tokens", type=int, default=205)
    parser.add_argument("--min_new_tokens", type=int, default=195)
    parser.add_argument("--num_test", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--sweet_threshold", type=float, default=0.6, help="Entropy threshold for SWEET")

    args = parser.parse_args()
    main(args)
