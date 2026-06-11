import os
from multiprocessing import Process, Queue
import json
from typing import List, Dict

import torch
from datasets import load_from_disk, load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from safetensors.torch import save_file
import nltk

from detection_utils import detect_lsh
from sampling_lsh_utils import lsh_reject_completion
from sampling_utils import extract_prompt_from_text
from sbert_lsh_model import SBERTLSHModel


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model name to embed sentences.
    embedder = None

    # Number of partitions in the embedding space. Default is 8.
    sp_dim: int = 3

    # Ratio of valid sentences.
    lmbd: float = 0.25

    # Logit augmentation for baseline or margin size for LSH and KMeans.
    delta = 0

    lsh_model = SBERTLSHModel(
        lsh_model_path=embedder, device=device, batch_size=1, lsh_dim=sp_dim, sbert_type='base'
    )


    watermarked_attacked_file = "/home/haojifei/dev_projects/nlp_projects/Learn-LSH/watermark-result/1300:1350_e:Qwen3-Embedding-8B_OpenAIPara.jsonl"
    with open(watermarked_attacked_file, 'r') as f:
        lines = f.readlines()

    file = open(watermarked_attacked_file, 'w')

    for item in tqdm(lines):
        l_json = json.loads(item)
        attackedTexts = l_json['text_attacked']['text_list']
        attacked_texts_results = []
        for attacked_text in attackedTexts:
            sentences = nltk.sent_tokenize(attacked_text)
            zScore = detect_lsh(
                sents=sentences, lsh_model=lsh_model,
                lmbd=lmbd, lsh_dim=sp_dim
            )
            attackedTextDetectionResult = {
                "is_watermarked": 1 if zScore > 4 else 0,
                "score": zScore
            }
            attacked_texts_results.append(attackedTextDetectionResult)
        l_json['text_attacked']['detect'] = attacked_texts_results
        file.write(json.dumps(l_json) + '\n')
    file.close()
