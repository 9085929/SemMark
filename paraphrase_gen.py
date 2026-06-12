import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None
from sampling_utils import extract_prompt_from_text
import os
import signal
import argparse
import random
import re
from typing import List
import logging
from concurrent.futures import ThreadPoolExecutor

import torch
from nltk import sent_tokenize
from tqdm import tqdm, trange
from datasets import load_from_disk, Dataset, DatasetDict
import openai
#from paraphrase_gen_utils import gen_prompt, query_openai#query_openai_bigram, gen_bigram_prompt, extract_list重述攻击
from paraphrase_gen_utils import gen_prompt, query_openai, gen_prompt_synonym # gen_prompt_synonym同义词替换攻击
#from paraphrase_gen_utils import gen_prompt, query_openai, gen_prompt_backtranslation_standard#回译攻击
#from paraphrase_gen_utils import gen_prompt, query_openai, gen_prompt_paper_regular, gen_prompt_pure_bt_step1, gen_prompt_pure_bt_step2, gen_prompt_enhanced_paraphrase, extract_list#回译攻击

def paraphrase_openai(client, texts: List[str], num_beams: int, bigram: bool = False):
    new_texts = []
    all_paras = []
    MAX_ITER = 10
    for text in tqdm(texts, desc="Paraphrasing with OpenAI"):
        sents = sent_tokenize(text)
        para_sents = []
        fail = False
        for i in range(len(sents)):
            sent = sents[i]
            context = sents[:i]
            num_iter = 0
            if bigram:
                pass
            else:
                prompt = gen_prompt(sent, context)
                para = query_openai(client, prompt)
                para_sents.append(para)
        if not fail:
            new_texts.append(sents)
            all_paras.append(para_sents)

    save_path = args.data_path + f'-openai-num_beams={num_beams}-bigram={bigram}'

    dataset = Dataset.from_dict({'text': new_texts, 'para_text': all_paras})
    train_test_split = dataset.train_test_split(test_size=0.3, seed=42)  
    test_valid = train_test_split['test'].train_test_split(test_size=0.333)  
    final_dataset = DatasetDict({
        'train': train_test_split['train'],  
        'valid': test_valid['train'],  
        'test': test_valid['test']  
    })
    final_dataset.save_to_disk(save_path)
    print(f"数据集已保存至：{save_path}")

    return new_texts, all_paras


def gen_prompt_v2(sent, context, num_paraphrases=5):

    prompt = (
        f"Previous context: {context}\n"
        f"Current sentence to paraphrase: {sent}\n"
        f"Please provide {num_paraphrases} diverse paraphrased versions of this sentence in a numbered list.\n"
        f"Strict Academic Requirements:\n"
        f"1. Semantic Equivalence: You MUST strictly preserve the original meaning, core facts, and specific entities. Do NOT add or delete information.\n"
        f"2. High Diversity: Rewrite the sentence using distinct vocabulary (synonyms) and different syntactic structures (e.g., alter the clause order, change active to passive) compared to the original.\n"
        f"3. Fluency: The paraphrased sentences must be grammatically correct and read naturally in the given context."
    )
    return prompt


def extract_list(text):
    if not text:
        return []
        
    pattern = r'\d+\.\s*(.*?)(?=\n\d+\.|\Z)'
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return [match.strip() for match in matches if match.strip()]


def random_mask_tokens(tokens: List[str], mask_rate: float, tokenizer):
    output = []
    for tok in tokens:
        if random.random() < mask_rate:
            output.append(tokenizer.mask_token)
        else:
            output.append(tok)
    return output


def paraphrase_openai_v2(
        client, texts: List[str],
        start_i: int, end_i: int,
        num_paraphrases: int = 8,
        max_loop: int = 10
):
    new_texts = []
    all_paras1 = []
    all_paras2 = []
    all_paras3 = []
    all_paras4 = []
    # all_paras5 = []
    # all_paras6 = []
    # all_paras7 = []
    # all_paras8 = []

    #save_path = args.data_path + f'-0{start_i}:0{end_i}-para{num_paraphrases}'
    #save_path = args.data_path + f'-0{start_i}:0{end_i}-para{num_paraphrases}-light'#重述攻击保存路径
    save_path = args.data_path + f'-0{start_i}:0{end_i}-para{num_paraphrases}-synonym'#同义词替换攻击保存路径
    #save_path = args.data_path + f'-0{start_i}:0{end_i}-para{num_paraphrases}-backtrans-std'#回译攻击保存路径
    
    logger.info(f'save_path: {save_path}')
    fail_texts_index = []
    consecutive_fail_count = 0  
    MAX_CONSECUTIVE_FAILS = 10
    for i, text in enumerate(texts):  # tqdm(texts, desc="Paraphrasing with OpenAI"):
        raw_text_str = " ".join(text) if isinstance(text, list) else text
        prompt = extract_prompt_from_text(raw_text_str, 32)
        gen_text = raw_text_str[len(prompt):].strip()
        
        sents = sent_tokenize(gen_text)
        # para_sents = []
        para_sents1 = []
        para_sents2 = []
        para_sents3 = []
        para_sents4 = []
        # para_sents5 = []
        # para_sents6 = []
        # para_sents7 = []
        # para_sents8 = []

        is_all_done = True
        logger.info(f"[{i}] --- text-length: {len(sents)} ---")
        for j in range(len(sents)):
            logger.info(f"[{i}][{j}]")
            curr_sent = sents[j]
            context = sents[:j]
            #gpt_prompt = gen_prompt_v2(curr_sent, context, num_paraphrases=num_paraphrases)#重述攻击调用
            gpt_prompt = gen_prompt_synonym(curr_sent, context, num_paraphrases=num_paraphrases)#同义词替换攻击调用
            #gpt_prompt = gen_prompt_backtranslation_standard(curr_sent, context, num_paraphrases=num_paraphrases)#回译攻击调用
            num_iter = 0
            para_ls = []

            while len(para_ls) < num_paraphrases and num_iter < max_loop:
                response = query_openai(client, gpt_prompt)
                para_ls = extract_list(response)
                logger.info(
                    f"getting [{i}][{j}] all paraphrased: {len(para_ls)} (should be {num_paraphrases} in total)"
                )
                num_iter += 1
                import time
                if len(para_ls) < num_paraphrases:
                    time.sleep(12)  
                else:
                    time.sleep(2)

            if len(para_ls) > 0:
                while len(para_ls) < 4:
                    para_ls.append(para_ls[-1])      
                para_sents1.append(para_ls[0])
                para_sents2.append(para_ls[1])
                para_sents3.append(para_ls[2])
                para_sents4.append(para_ls[3])
            else:
                is_all_done = False
                logger.warning(f"jump sentence: [{i}] (一句都没提取到)")
                fail_texts_index.append(i)
                break

        if is_all_done:
            consecutive_fail_count = 0  
            new_texts.append([prompt] + sents)  
            all_paras1.append([prompt] + para_sents1)
            all_paras2.append([prompt] + para_sents2)
            all_paras3.append([prompt] + para_sents3)
            all_paras4.append([prompt] + para_sents4)
        else:
            consecutive_fail_count += 1
            logger.warning(f"🚨 当前连续失败次数: {consecutive_fail_count}/{MAX_CONSECUTIVE_FAILS}")
            
            if consecutive_fail_count >= MAX_CONSECUTIVE_FAILS:
                logger.error(f"❌ 连续 {MAX_CONSECUTIVE_FAILS} 次失败！程序即将挂起 (类似 Ctrl+Z)。")
                logger.info("👉 修复问题后，在终端输入 'fg' 命令即可恢复运行...")                
                os.kill(os.getpid(), signal.SIGTSTP)
                logger.info("▶️ 进程已恢复！重置失败计数器，继续执行...")
                consecutive_fail_count = 0

    dataset = Dataset.from_dict({
        'text': new_texts,
        'para_text1': all_paras1,
        'para_text2': all_paras2,
        'para_text3': all_paras3,
        'para_text4': all_paras4,
    })

    logger.info(f'dataset info: \n{dataset}')

    train_test_split = dataset.train_test_split(test_size=0.3, seed=42)
    test_valid = train_test_split['test'].train_test_split(test_size=0.333)

    final_dataset = DatasetDict({
        'train': train_test_split['train'],
        'valid': test_valid['train'],
        'test': test_valid['test']
    })

    final_dataset.save_to_disk(save_path)
    logger.info(f"数据集已保存至：{save_path}")
    logger.info(f"释义失败的索引: {fail_texts_index}")

    return new_texts, 1

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

def paraphrase_pegasus_clean(texts, local_model_path, start_i, end_i, num_beams=4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"正在从本地加载 Pegasus: {local_model_path}...")
    model = PegasusForConditionalGeneration.from_pretrained(local_model_path).to(device)
    tokenizer = PegasusTokenizer.from_pretrained(local_model_path)

    new_texts = []
    all_paras1 = []

    for text in tqdm(texts, desc="Pegasus 洗稿中"):
        sents = sent_tokenize(text)
        para_sents1 = []
        
        for sent in sents:
            batch = tokenizer(sent, truncation=True, padding='longest', return_tensors="pt", max_length=60).to(device)
            translated = model.generate(**batch, max_length=60, num_beams=num_beams, num_return_sequences=1)
            tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
            para_sents1.append(tgt_text[0])

        new_texts.append(sents)
        all_paras1.append(para_sents1)

    # 构建与 OpenAI 版本结构一致的 Dataset
    dataset = Dataset.from_dict({
        'text': new_texts,
        'para_text1': all_paras1,
    })

    # 切分数据集 7:2:1 保持对齐
    train_test_split = dataset.train_test_split(test_size=0.3, seed=42)
    test_valid = train_test_split['test'].train_test_split(test_size=0.333)
    final_dataset = DatasetDict({
        'train': train_test_split['train'],
        'valid': test_valid['train'],
        'test': test_valid['test']
    })

    save_path = args.data_path + f'-pegasus-{start_i}_{end_i}'
    final_dataset.save_to_disk(save_path)
    logger.info(f"Pegasus 数据集已安全保存至：{save_path}")
def paraphrase_openai_baseline(
        client, texts: List[str],
        start_i: int, end_i: int,
        max_loop: int = 3
):
    new_texts = []
    all_paras1 = [] 
    
    save_path = args.data_path + f'-0{start_i}:0{end_i}-para1-paper-baseline'
    logger.info(f'save_path: {save_path}')
    fail_texts_index = []

    for i, text in enumerate(texts):
        if isinstance(text, list):
            sents = text
        else:
            sents = sent_tokenize(text)
            
        para_sents1 = []
        is_all_done = True
        
        logger.info(f"[{i}] --- text-length: {len(sents)} ---")
        for j in range(len(sents)):
            curr_sent = sents[j]
            context = sents[:j]
            
            prompt = gen_prompt_paper_regular(curr_sent, context)
            num_iter = 0
            para = ""

            while not para and num_iter < max_loop:
                response = query_openai(client, prompt)
                if response:
                    para = response.strip().split('\n')[0].strip('\"\'')
                num_iter += 1

            if para:
                para_sents1.append(para)
            else:
                is_all_done = False
                logger.warning(f"jump sentence: [{i}]")
                fail_texts_index.append(i)
                break

        if is_all_done:
            new_texts.append(sents)
            all_paras1.append(para_sents1)

    dataset = Dataset.from_dict({
        'text': new_texts,
        'para_text1': all_paras1,
    })

    train_test_split = dataset.train_test_split(test_size=0.3, seed=42)
    test_valid = train_test_split['test'].train_test_split(test_size=0.333)

    final_dataset = DatasetDict({
        'train': train_test_split['train'],
        'valid': test_valid['train'],
        'test': test_valid['test']
    })

    final_dataset.save_to_disk(save_path)
    logger.info(f"Baseline 数据集已保存至：{save_path}")
    if fail_texts_index:
        logger.info(f"释义失败的索引: {fail_texts_index}")

    return new_texts, 1
def process_single_sentence_bt(curr_sent, client, max_loop):
    num_iter = 0
    final_english = ""
    while not final_english and num_iter < max_loop:
        # Step 1: English -> French
        prompt_step1 = gen_prompt_pure_bt_step1(curr_sent)
        french_sent = query_openai(client, prompt_step1, temperature=0.0)
        
        if french_sent:
            # Step 2: French -> English
            prompt_step2 = gen_prompt_pure_bt_step2(french_sent)
            english_sent = query_openai(client, prompt_step2, temperature=0.0)
            if english_sent:
                final_english = english_sent.strip().strip('\"\'')
        
        num_iter += 1
    return final_english

def paraphrase_openai_pure_bt(
        client, texts: List[str],
        start_i: int, end_i: int,
        max_loop: int = 3
):
    new_texts = []
    all_paras1 = [] 
    
    save_path = args.data_path + f'-0{start_i}:0{end_i}-pure-backtranslation'
    logger.info(f'save_path: {save_path}')
    fail_texts_index = []
    MAX_WORKERS = 10 

    for i, text in enumerate(texts):
        raw_text_str = " ".join(text) if isinstance(text, list) else text
        prompt = extract_prompt_from_text(raw_text_str, 32)
        gen_text = raw_text_str[len(prompt):].strip()
        
        sents = sent_tokenize(gen_text)
        para_sents1 = []
        is_all_done = True
        
        logger.info(f"[{i}] --- text-length: {len(sents)} --- 开始并发请求...")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_single_sentence_bt, sent, client, max_loop) for sent in sents]
            
            for future in futures:
                result = future.result()
                if result:
                    para_sents1.append(result)
                else:
                    is_all_done = False
                    break

        if is_all_done:
            new_texts.append([prompt] + sents)
            all_paras1.append([prompt] + para_sents1)
            logger.info(f"[{i}] 处理完成！")
        else:
            logger.warning(f"[{i}] 有句子连续 {max_loop} 次请求失败，跳过该段落。")
            fail_texts_index.append(i)

    dataset = Dataset.from_dict({
        'text': new_texts,
        'para_text1': all_paras1,
    })

    train_test_split = dataset.train_test_split(test_size=0.3, seed=42)
    test_valid = train_test_split['test'].train_test_split(test_size=0.333)

    final_dataset = DatasetDict({
        'train': train_test_split['train'],
        'valid': test_valid['train'],
        'test': test_valid['test']
    })

    final_dataset.save_to_disk(save_path)
    logger.info(f"Pure Back-translation 数据集已保存至：{save_path}")
    if fail_texts_index:
        logger.info(f"释义失败的索引: {fail_texts_index}")

    return new_texts, 1
if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s: %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(f'{__file__}.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_path', type=str, default='AbeHou/opt-1.3b-semstamp')
    parser.add_argument('--bsz', type=int, default=-1)
    parser.add_argument(
        '--paraphraser', type=str, default="openai", help='paraphraser type',
        choices=[
            'pegasus',
            'parrot',
            'openai',
            'parrot-bigram',
            'pegasus-bigram',
            'openai-bigram',
            'openai-baseline',
            'openai-pure-bt',          
            'openai-enhanced-attack'
        ]
    )
    parser.add_argument('--temp', type=float, default=2.0, help='decode temperature')
    parser.add_argument(
        '--bert_threshold', type=float, default=0.0,
        help='threshold for bert similarity between original and paraphrased'
    )
    parser.add_argument('--num_beams', type=int, default=4, help='number of beams for beam-search')
    parser.add_argument('--max_iter', type=int, default=8, help='最大重新生成')
    parser.add_argument('--start_i', type=int, default=0)
    parser.add_argument('--end_i', type=int, default=-1)
    args = parser.parse_args()
    logger.info(f"args:{args}")

    tokenizer = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = load_from_disk(args.data_path)
    logger.info(f"dataset info: {dataset}")
    from datasets import DatasetDict, concatenate_datasets
    if isinstance(dataset, DatasetDict):
        dataset = concatenate_datasets(list(dataset.values()))
        logger.info(f"检测到 DatasetDict，已自动合并。合并后总数据量: {len(dataset)}")

    start_i = args.start_i
    end_i = len(dataset) if args.end_i == -1 else args.end_i
    
    logger.info(f"===> 真正执行切片: start_i: {start_i}, end_i: {end_i} <===")
    texts = dataset['text'][start_i:end_i]
    del dataset

    if args.paraphraser == 'openai':
        client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL')
        )
        new_texts, paras = paraphrase_openai_v2(
            client, texts, start_i, end_i, num_paraphrases=args.num_beams, max_loop=args.max_iter
        )
    elif args.paraphraser == 'pegasus':
        local_path = "/home/haojifei/dev_resource/huggingface/models/tuner007/pegasus_paraphrase"
        paraphrase_pegasus_clean(texts, local_path, start_i, end_i, num_beams=args.num_beams)
        
    elif args.paraphraser == 'openai-bigram':
        client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL')
        )
        new_texts, paras = paraphrase_openai(client, texts, args.num_beams, bigram=True)
    elif args.paraphraser == 'openai-baseline':
        client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL')
        )
        new_texts, paras = paraphrase_openai_baseline(
            client, texts, start_i, end_i, max_loop=args.max_iter
        )
    elif args.paraphraser == 'openai-pure-bt':
        client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL')
        )
        new_texts, paras = paraphrase_openai_pure_bt(
            client, texts, start_i, end_i, max_loop=args.max_iter
        )
        
    elif args.paraphraser == 'openai-enhanced-attack':
        client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL')
        )
        new_texts, paras = paraphrase_openai_v2(
            client, texts, start_i, end_i, num_paraphrases=args.num_beams, max_loop=args.max_iter
        )
    else:
        raise NotImplementedError
