import pprint
import argparse
import os
import sys
import torch
import time
import torch.multiprocessing as mp
from datasets import load_from_disk, Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from sbert_lsh_model import SBERTLSHModel
from sentence_transformers import SentenceTransformer
from multiprocessing import Process, Queue
import numpy as np
from nltk.tokenize import sent_tokenize
from samplingg_utils import extract_prompt_from_text
from sampling_lsh_utils import lsh_reject_completion
from sampling_semstamp_official import semstamp_official_completion
from sampling_kmeans_utils import kmeans_reject_completion, load_embeds
import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None
from transformers import BitsAndBytesConfig
from datasets import concatenate_datasets
PUNCTS = '.,!?'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data', type=str,
        help='Path to Hugging Face dataset that has a column "text".'
    )
    parser.add_argument(
        '--model', type=str,
        help='Model name to generate continuation. HuggingFace/OpenAI.',
        default="facebook/opt-1.3b"
    )
    parser.add_argument(
        '--embedder', type=str,
        help='Model name to embed sentences.', default=None
    )
    parser.add_argument(
        '--len_prompt', '-l', type=int, default=32,
        help='MAX length of prompt.'
    )
    parser.add_argument(
        '--max_new_tokens', type=int, default=205,
        help='Maximum number of new tokens to generate.'
    )
    parser.add_argument(
        '--min_new_tokens', type=int, default=195,
        help='Minimum number of new tokens to generate.'
    )
    parser.add_argument(
        '--rep_p', type=float, default=1.05,
        help='Repetition penalty.'
    )
    parser.add_argument(
        '--lmbd', type=float, default=0.5,
        help='Ratio of valid sentences.'
    )
    parser.add_argument(
        '--delta', type=float, default=0,
        help='Logit augmentation for baseline or margin size for LSH and KMeans.'
    )
    parser.add_argument(
        '--sp_mode', type=str, default='lsh', 
        choices=['none', 'kmeans', 'lsh', 'semstamp_official'], 
        help='Sentence prompt mode.'
    )
    parser.add_argument(
        '--sp_dim', type=int, default=3,
        help='Number of partitions in the embedding space. Default is 8.'
    )
    parser.add_argument(
        '--embed_path', type=str,
        help='Path to precomputed embed for training KMeans.', default=None
    )
    parser.add_argument(
        '--cc_path', type=str,
        help='KMeans precomputed cluster centers data.', default=None
    )
    parser.add_argument(
        '--sweet_threshold', type=float, default=0.6,
        help='Entropy threshold for SWEET generating filter.'
    )
    pp = pprint.PrettyPrinter(indent=4)
    args = parser.parse_args()
    pp.pprint(vars(args))  # Debug print for parsed arguments
    return args


def worker(rank, dataset_chunk, output_queue, args, device):

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        quantization_config=quantization_config,
        device_map={"": device},  
        trust_remote_code=True
    )
    
    model.eval()


    if args.sp_mode == "lsh":
        gen_config = GenerationConfig(
            max_new_tokens=250, 
            min_new_tokens=190, 
            do_sample=True,
            temperature=0.7,
            top_k=0,
            repetition_penalty=args.rep_p,
        )
    else:

        gen_config = GenerationConfig(
            max_new_tokens=210, 
            do_sample=True,
            temperature=0.7,
            top_k=0,
            repetition_penalty=args.rep_p,
        )

    if args.sp_mode == "lsh":
        lsh_model = SBERTLSHModel(
            lsh_model_path=args.embedder, device=device, batch_size=1, lsh_dim=args.sp_dim, sbert_type='base'
        )


        def text_to_generated_text(ex):
            raw_text = " ".join(ex['text']) if isinstance(ex['text'], list) else ex['text']
            prompt = extract_prompt_from_text(raw_text, args.len_prompt)
            
            torch.cuda.synchronize(device=device)
            start_time = time.perf_counter()
            response = lsh_reject_completion(
                prompt, model, tokenizer, gen_config, lsh_model, args.sp_dim,
                lmbd=args.lmbd, device=device, margin=args.delta,
                sweet_threshold=args.sweet_threshold  
            )
            
            torch.cuda.synchronize(device=device)
            end_time = time.perf_counter()
            
            elapsed_time = end_time - start_time
            new_text = response[len(prompt):].strip()
            generated_sents = sent_tokenize(new_text)
            num_sents = len(generated_sents)
            num_tokens = len(tokenizer.encode(new_text, add_special_tokens=False))

            ex['elapsed_time'] = elapsed_time
            ex['num_sents'] = num_sents
            ex['num_tokens'] = num_tokens
            ex['sec_per_sent'] = elapsed_time / num_sents if num_sents > 0 else 0
            ex['tokens_per_sec'] = num_tokens / elapsed_time if elapsed_time > 0 else 0

            ex['text'] = prompt + " " + new_text
            return ex
    elif args.sp_mode == 'semstamp_official':
        lsh_model = SBERTLSHModel(
            lsh_model_path=args.embedder, device=device, batch_size=1, lsh_dim=args.sp_dim, sbert_type='base'
        )

        def text_to_generated_text(ex):
            raw_text = " ".join(ex['text']) if isinstance(ex['text'], list) else ex['text']
            prompt = extract_prompt_from_text(raw_text, args.len_prompt)

            torch.cuda.synchronize(device=device)
            start_time = time.perf_counter()
            
            response = semstamp_official_completion(
                prompt, model, tokenizer, gen_config, lsh_model, args.sp_dim,
                lmbd=args.lmbd, device=device, margin=args.delta
            )

            torch.cuda.synchronize(device=device)
            end_time = time.perf_counter()
            
            elapsed_time = end_time - start_time
            if isinstance(response, tuple):
                new_text = response[0]
            else:
                new_text = response
                
            new_text_str = new_text.strip() if isinstance(new_text, str) else new_text
            if new_text_str.startswith(prompt):
                gen_only = new_text_str[len(prompt):].strip()
            else:
                gen_only = new_text_str.strip()
                new_text_str = prompt + " " + gen_only
                
            generated_sents = sent_tokenize(gen_only)
            num_sents = len(generated_sents)
            num_tokens = len(tokenizer.encode(gen_only, add_special_tokens=False))
            ex['elapsed_time'] = elapsed_time
            ex['num_sents'] = num_sents
            ex['num_tokens'] = num_tokens
            ex['sec_per_sent'] = elapsed_time / num_sents if num_sents > 0 else 0
            ex['tokens_per_sec'] = num_tokens / elapsed_time if elapsed_time > 0 else 0
                
            ex['text'] = new_text_str
            return ex
    elif args.sp_mode == "kmeans":
        cluster_centers = torch.load(args.cc_path)
        # print(f"Load cluster centers: {cluster_centers.shape}")

        embedder = SentenceTransformer(args.embedder, device=device)

        def text_to_generated_text(ex):
            raw_text = " ".join(ex['text']) if isinstance(ex['text'], list) else ex['text']
            prompt = extract_prompt_from_text(raw_text, args.len_prompt)
            torch.cuda.synchronize(device=device)
            start_time = time.perf_counter()
            
            response = kmeans_reject_completion(
                prompt=prompt, model=model, tokenizer=tokenizer, gen_config=gen_config, embedder=embedder,
                cluster_centers=cluster_centers, lmbd=args.lmbd, k_dim=args.sp_dim, margin=args.delta, device=device
            )
            torch.cuda.synchronize(device=device)
            end_time = time.perf_counter()
            
            elapsed_time = end_time - start_time
            if isinstance(response, tuple):
                new_text = response[0]
            else:
                new_text = response
                
            new_text_str = new_text.strip() if isinstance(new_text, str) else new_text
            if new_text_str.startswith(prompt):
                gen_only = new_text_str[len(prompt):].strip()
            else:
                gen_only = new_text_str.strip()
                new_text_str = prompt + " " + gen_only
                
            generated_sents = sent_tokenize(gen_only)
            num_sents = len(generated_sents)
            num_tokens = len(tokenizer.encode(gen_only, add_special_tokens=False))
            ex['elapsed_time'] = elapsed_time
            ex['num_sents'] = num_sents
            ex['num_tokens'] = num_tokens
            ex['sec_per_sent'] = elapsed_time / num_sents if num_sents > 0 else 0
            ex['tokens_per_sec'] = num_tokens / elapsed_time if elapsed_time > 0 else 0
            
            ex['text'] = new_text_str
            return ex
    else:
        raise NotImplementedError

    processed_chunk = dataset_chunk.map(text_to_generated_text, batch_size=1)
    output_queue.put(processed_chunk)


def parallel_generate(args):
    """
    Splits the dataset and distributes work across multiple GPUs by index.
    """
    dataset = load_from_disk(args.data)
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected. This script requires at least one GPU.")

    print(f"Detected {num_gpus} GPU(s). Splitting dataset for parallel processing.")

    output_queue = Queue()
    processes = []

    # Create a process for each GPU and its respective dataset shard
    for rank in range(num_gpus):
        device = f"cuda:{rank}"
        # Shard the dataset by assigning a specific index to the GPU
        dataset_chunk = dataset.shard(num_shards=num_gpus, index=rank)
        p = Process(target=worker, args=(rank, dataset_chunk, output_queue, args, device))
        p.start()
        processes.append(p)

    all_results = []
    for _ in processes:
        all_results.append(output_queue.get())

    for p in processes:
        p.join()
    
    merged_dataset = concatenate_datasets(all_results)
    
    embedder_name = args.embedder.split('/')[-1] if args.embedder else "default"
    output_path = os.path.join(args.data, f"{args.sp_mode}-{embedder_name}-sweet{args.sweet_threshold}")
    os.makedirs(output_path, exist_ok=True)
    merged_dataset.save_to_disk(output_path)
    avg_sec_per_sent = sum(merged_dataset['sec_per_sent']) / len(merged_dataset)
    avg_tps = sum(merged_dataset['tokens_per_sec']) / len(merged_dataset)
    print(f"\n📊 评测完成！平均生成速度: {avg_sec_per_sent:.3f} 秒/句, 吞吐量: {avg_tps:.1f} Tokens/s")


if __name__ == '__main__':
    args = parse_args()
    mp.set_start_method('spawn', force=True)
    parallel_generate(args)
