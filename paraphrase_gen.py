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
# import pickle
from concurrent.futures import ThreadPoolExecutor

import torch
from nltk import sent_tokenize
from tqdm import tqdm, trange
# from transformers import AutoTokenizer
# from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from datasets import load_from_disk, Dataset, DatasetDict
import openai
# from dipper import DipperParaphraser
# from paraphrase_gen_utils import accept_by_bigram_overlap, SParrot
#from paraphrase_gen_utils import gen_prompt, query_openai#query_openai_bigram, gen_bigram_prompt, extract_list重述攻击
from paraphrase_gen_utils import gen_prompt, query_openai, gen_prompt_synonym # gen_prompt_synonym同义词替换攻击
#from paraphrase_gen_utils import gen_prompt, query_openai, gen_prompt_backtranslation_standard#回译攻击
#from paraphrase_gen_utils import gen_prompt, query_openai, gen_prompt_paper_regular, gen_prompt_pure_bt_step1, gen_prompt_pure_bt_step2, gen_prompt_enhanced_paraphrase, extract_list
# from sampling_utils import well_formed_sentence

# device = 'cuda' if torch.cuda.is_available() else "cpu"
# num_beams = 25


# def pegasus_paraphrase(
#         texts, tokenizer,
#         paraphraser_name="tuner007/pegasus_paraphrase",
#         device='cuda', num_beams=10,
#         temp=2, bsz=-1, bigram=False, bert_threshold=0.03
# ):
#     paraphraser = PegasusForConditionalGeneration.from_pretrained(paraphraser_name).to(device)
#     paraphraser_tokenizer = PegasusTokenizer.from_pretrained(paraphraser_name)
#
#     def paraphrase(sents):
#         '''
#         Arguments:
#             sents: list of sentences (max len under 60!)
#         Returns:
#             paraphrased: list of paraphrased sents
#         '''
#         batch = paraphraser_tokenizer(
#             sents, truncation=True, padding='longest', return_tensors="pt", max_length=60).to(device)
#
#         paraphrased_ids = paraphraser.generate(
#             **batch, max_length=60, num_beams=num_beams, num_return_sequences=num_beams, temperature=temp,
#             repetition_penalty=1.03)
#         # batch decode and return the first one
#         paraphrased = [paraphraser_tokenizer.decode(paraphrased_ids[i * num_beams], skip_special_tokens=True) for i in
#                        range(len(paraphrased_ids) // num_beams)]
#         # breakpoint()
#
#         return paraphrased
#
#     # dataset has to be a text list
#     sents, data_len = [], []
#     for text in tqdm(texts, desc="Tokenizer"):
#         sent_list = sent_tokenize(text)
#         sents.extend(sent_list)
#         data_len.append(len(sent_list))
#     paras = []
#
#     if bsz != -1:
#         batched_sents = [sents[i:i + bsz] for i in range(0, len(sents), bsz)]
#         for batch in tqdm(batched_sents, desc="Paraphrasing"):
#             paraphrased = paraphrase(batch)
#             paras.extend(paraphrased)
#
#     else:
#         for sent in tqdm(sents):
#             paraphrased = paraphrase([sent])
#             paraphrased = [well_formed_sentence(para) for para in paraphrased]
#             if bigram:
#                 para = accept_by_bigram_overlap(sent, paraphrased, tokenizer, bert_threshold)
#             else:
#                 para = paraphrased[0]
#             paras.append(para)
#
#     start_pos = 0
#     output = []
#     new_texts = []
#     for l in data_len:
#         output.append(paras[start_pos: start_pos + l])
#         new_texts.append(sents[start_pos: start_pos + l])
#         start_pos += l
#     new_dataset = Dataset.from_dict(
#         {'text': new_texts, 'para_text': output})
#     name = args.data_path + \
#            f'-pegasus-bigram={bigram}-threshold={bert_threshold}'
#     new_dataset.save_to_disk(name)
#     return output
#
#
# def parrot_paraphrase(
#         parrot, texts, tokenizer, num_beams=10, bigram=False, save_to_disk=True, avg_sent_len=20,
#         save_by_sents=False, bert_threshold=0.03
# ):
#     # modified parrot source code to have the num_beams argument
#     def paraphrase(sent):
#         para_phrases = parrot.augment(
#             input_phrase=sent,
#             use_gpu=True,
#             diversity_ranker="levenshtein",
#             do_diverse=True,
#             max_return_phrases=num_beams,
#             max_length=60,
#             adequacy_threshold=0.8,
#             fluency_threshold=0.8
#         )
#         return para_phrases
#
#     sents, data_len = [], []
#     for text in tqdm(texts, desc="Tokenizer"):
#         sent_list = sent_tokenize(text)
#         sents.extend(sent_list)
#         data_len.append(len(sent_list))
#     start_pos = 0
#     paras = []
#     total_paraphrased = []
#     for sent in tqdm(sents):
#         paraphrased = paraphrase(sent)
#         paraphrased = [well_formed_sentence(
#             para, end_sent=True) for para in paraphrased]
#         total_paraphrased.append(paraphrased)
#         if bigram:
#             para = accept_by_bigram_overlap(sent, paraphrased, tokenizer, bert_threshold=bert_threshold)
#         paras.append(para)
#     start_pos = 0
#     output = []
#     new_texts = []
#     if save_by_sents:
#         for l in data_len:
#             output.append(paras[start_pos: start_pos + l])
#             new_texts.append(sents[start_pos: start_pos + l])
#             start_pos += l
#     elif save_to_disk:
#         new_texts = texts
#         for l in data_len:
#             output.append(" ".join(paras[start_pos: start_pos + l]))
#             start_pos += l
#     new_dataset = Dataset.from_dict({'text': new_texts, 'para_text': output})
#     name = args.data_path + \
#            f'-parrot-bigram={bigram}-threshold={bert_threshold}'
#     new_dataset.save_to_disk(name)
#     pkl_name = args.data_path + f'-parrot-bigram={bigram}-threshold={bert_threshold}-all_beams.pkl'
#     with open(pkl_name, 'wb') as f:
#         pickle.dump(total_paraphrased, f)
#         f.close()
#     return output


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
                # para_ls = []
                # prompt = gen_bigram_prompt(sent, context, num_beams)
                # # if insufficient number of para_sents generated, try again
                # while (len(para_ls) < 5 and num_iter < MAX_ITER):
                #     para_str = query_openai_bigram(client, prompt)
                #     # use regex to extract list from string
                #     para_ls = extract_list(para_str)
                #     num_iter += 1
                #     # openai refuses to paraphrase, thendiscard
                # if num_iter <= MAX_ITER:
                #     para_sents.append(para_ls)
                # else:
                #     fail = True
            else:
                prompt = gen_prompt(sent, context)
                para = query_openai(client, prompt)
                para_sents.append(para)
        if not fail:
            new_texts.append(sents)
            all_paras.append(para_sents)

    save_path = args.data_path + f'-openai-num_beams={num_beams}-bigram={bigram}'
    # Dataset.from_dict({'text': new_texts, 'para_text': all_paras}).save_to_disk(save_path)

    # 构建基础 Dataset
    dataset = Dataset.from_dict({'text': new_texts, 'para_text': all_paras})
    # 划分数据集：7:2:1 → train:70%, validation:20%, test:10%
    train_test_split = dataset.train_test_split(test_size=0.3, seed=42)  # 先切出 30% 做 test+valid
    test_valid = train_test_split['test'].train_test_split(test_size=0.333)  # 0.3 * 0.333 ≈ 0.1 → test=10%
    # 构造 DatasetDict
    final_dataset = DatasetDict({
        'train': train_test_split['train'],  # 70%
        'valid': test_valid['train'],  # 约 20%
        'test': test_valid['test']  # 约 10%
    })
    # 保存到磁盘
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
    """从 GPT 输出中提取列表项"""
    # 如果 text 是 None 或空字符串，直接返回空列表
    if not text:
        return []
        
    pattern = r'\d+\.\s*(.*?)(?=\n\d+\.|\Z)'
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return [match.strip() for match in matches if match.strip()]


def random_mask_tokens(tokens: List[str], mask_rate: float, tokenizer):
    """随机mask部分token"""
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
    # all_paras = []
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
                
                # 👇 关键代码：向当前进程发送 SIGTSTP 信号，模拟 Ctrl+Z
                os.kill(os.getpid(), signal.SIGTSTP)
                
                # 挂起恢复后，重置计数器并继续循环
                logger.info("▶️ 进程已恢复！重置失败计数器，继续执行...")
                consecutive_fail_count = 0

    # 构建基础 Dataset
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
            # 使用 Pegasus 官方推荐的重述生成参数
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
# python paraphrase_gen.py --data_path data/c4-train-8000
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

    # 构建基础 Dataset (只有 para_text1)
    dataset = Dataset.from_dict({
        'text': new_texts,
        'para_text1': all_paras1,
    })

    # 划分数据集 7:2:1
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
    """用于多线程处理单句纯回译的辅助函数"""
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
        # 1. 剥离 Prompt
        raw_text_str = " ".join(text) if isinstance(text, list) else text
        prompt = extract_prompt_from_text(raw_text_str, 32)
        gen_text = raw_text_str[len(prompt):].strip()
        
        # 2. 只切分生成文本
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
            # 3. 拼回 Prompt
            new_texts.append([prompt] + sents)
            all_paras1.append([prompt] + para_sents1)
            logger.info(f"[{i}] 处理完成！")
        else:
            logger.warning(f"[{i}] 有句子连续 {max_loop} 次请求失败，跳过该段落。")
            fail_texts_index.append(i)


    # === 下方的数据集保存逻辑保持不变 ===
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
    # 配置 logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # 创建一个 formatter（格式器）
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s: %(message)s')

    # 输出到控制台的 Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 输出到文件的 Handler
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
            'openai-pure-bt',          # <=== 新增：纯回译基线
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

    # if args.paraphraser == 'parrot':
    #     parrot = SParrot()
    #     parrot_paraphrase(
    #         parrot, texts, tokenizer, num_beams=args.num_beams,
    #         bert_threshold=args.bert_threshold
    #     )
    # elif args.paraphraser == 'parrot-bigram':
    #     parrot = SParrot()
    #     parrot_paraphrase(
    #         parrot, texts, tokenizer, num_beams=args.num_beams,
    #         bert_threshold=args.bert_threshold, bigram=True
    #     )
    # elif args.paraphraser == 'pegasus-bigram':
    #     pegasus_paraphrase(
    #         texts, tokenizer, num_beams=args.num_beams, bert_threshold=args.bert_threshold, bigram=True
    #     )
    # elif args.paraphraser == 'pegasus':
    #     pegasus_paraphrase(
    #         texts, tokenizer, num_beams=args.num_beams, bert_threshold=args.bert_threshold, bsz=args.bsz
    #     )
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
        # === 新增：学术纯回译基线 ===
    elif args.paraphraser == 'openai-pure-bt':
        client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL')
        )
        new_texts, paras = paraphrase_openai_pure_bt(
            client, texts, start_i, end_i, max_loop=args.max_iter
        )
        
    # === 你原来的：增强型上下文攻击 ===
    elif args.paraphraser == 'openai-enhanced-attack':
        client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL')
        )
        # 注意：这里需要把你原来 paraphrase_openai_v2 里调用的 prompt 换成 gen_prompt_enhanced_paraphrase
        new_texts, paras = paraphrase_openai_v2(
            client, texts, start_i, end_i, num_paraphrases=args.num_beams, max_loop=args.max_iter
        )
    else:
        raise NotImplementedError
