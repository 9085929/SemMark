# 这个脚本是用来生成 watermarked-file（使用 SemStamp 水印方法） 的。

import sys
import logging
from pathlib import Path
import json
from typing import List, Dict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import nltk

from detection_utils import detect_lsh
from sampling_lsh_utils import lsh_reject_completion
from sampling_utils import extract_prompt_from_text
from sbert_lsh_model import SBERTLSHModel

# Logging 基本配置
LOG_FILE = Path(__file__).with_name("main.log")
logging.basicConfig(
    level=logging.DEBUG,  # 默认等级
    format="%(asctime)s-[%(levelname)s]-%(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # 控制台输出
        logging.FileHandler(LOG_FILE, encoding="utf-8")  # 文件输出
    ]
)


def load_json_dataset(path: str) -> List[Dict]:
    with open(path, 'r') as f:
        lines = f.readlines()
    d: List[dict] = []
    for line in lines:
        item = json.loads(line)
        d.append(item)
    return d


def text_to_generated_text(ex):
    prompt = extract_prompt_from_text(ex['prompt'], len_prompt)
    response = lsh_reject_completion(
        prompt, model, tokenizer, gen_config, lsh_model, sp_dim,
        lmbd=lmbd, device=device, margin=delta
    )
    ex['prompt'] = response.strip()
    return ex


if __name__ == '__main__':
    logger = logging.getLogger(__name__)  # 模块名作为 logger 名

    # Path to Hugging Face dataset that has a column "text".
    dataset_path: str = "/home/haojifei/dev_projects/nlp_projects/Learn-LSH/dataset/c4/processed_c4.jsonl"
    dataset_start: int = 0
    dataset_end: int = 2

    # MAX length of prompt.
    len_prompt: int = 32

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # "/home/haojifei/dev_resource/huggingface/models/facebook/opt-1.3b"
    # "/home/haojifei/dev_resource/huggingface/models/Qwen/Qwen3-8B"
    model_path: str = "/home/haojifei/dev_resource/huggingface/models/facebook/opt-1.3b"

    # Model name to generate continuation. HuggingFace or OpenAI.
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map='auto')  # , use_safetensors=False
    # state_dict = model.state_dict()
    # save_file(state_dict, "/home/haojifei/dev_resource/huggingface/models/facebook/opt-1.3b/model.safetensors")
    # model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Maximum number of new tokens to generate.
    max_new_tokens: int = 205
    # Minimum number of new tokens to generate.
    min_new_tokens: int = 195
    # Repetition penalty.
    rep_p: float = 1.05
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        # min_new_tokens=min_new_tokens,
        do_sample=True,
        temperature=0.7,
        # TODO top_k=0,
        repetition_penalty=rep_p,
    )

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

    # dataset = load_from_disk(dataset_path)
    # dataset = dataset['test']
    dataset = load_json_dataset(dataset_path)
    dstart: int = 1300
    dend: int = 1350
    dataset_start_end = "{:04d}:{:04d}".format(dstart, dend)
    dataset = dataset[dstart:dend]
    logger.info(f"loading dataset ({dataset_start_end})")

    output_file_path = Path(f"./watermark-result/SemStamp-genQwen/{dataset_start_end}_e:Qwen3-Embedding-0.6B_watermarked.jsonl")
    # 获取父目录（即不包含文件名的路径）
    parent_dir = output_file_path.parent
    # 创建目录（如果不存在），exist_ok=True 表示目录已存在时不报错
    parent_dir.mkdir(parents=True, exist_ok=True)
    # 可选：创建空的 jsonl 文件
    output_file_path.touch(exist_ok=True)  # 如果文件已存在，不会覆盖

    output_file = open(output_file_path, "w")

    for item in tqdm(dataset):
        prompt: str = extract_prompt_from_text(item['prompt'] + item['natural_text'], len_prompt)

        watermarked: str = lsh_reject_completion(
            prompt, model, tokenizer, gen_config, lsh_model, sp_dim,
            lmbd=lmbd, device=device, margin=delta
        )
        watermarked_sentences = nltk.sent_tokenize(watermarked)
        watermarked_detect_z_score = detect_lsh(
            sents=watermarked_sentences, lsh_model=lsh_model,
            lmbd=lmbd, lsh_dim=sp_dim
        )
        watermarked_is_watermarked: bool = watermarked_detect_z_score > 4
        watermarked_detect = {"is_watermarked": str(watermarked_is_watermarked), "score": watermarked_detect_z_score}

        unwatermarked = item['prompt'] + item['natural_text']
        unwatermarked_sentences = nltk.sent_tokenize(unwatermarked)
        unwatermarked_detect_z_score = detect_lsh(
            sents=unwatermarked_sentences, lsh_model=lsh_model,
            lmbd=lmbd, lsh_dim=sp_dim
        )
        unwatermarked_text_is_watermarked: bool = unwatermarked_detect_z_score > 4
        unwatermarked_text_detect = {
            "is_watermarked": str(unwatermarked_text_is_watermarked),
            "score": unwatermarked_detect_z_score
        }

        data_dict = {
            'prompt': prompt,
            'watermarked': watermarked,
            'watermarked_detect': watermarked_detect,
            'unwatermarked': unwatermarked,
            'unwatermarked_detect': unwatermarked_text_detect,
        }

        data_json = json.dumps(data_dict)
        output_file.write(data_json + '\n')
        output_file.flush()
