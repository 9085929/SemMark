import os
import argparse
import random
import re
from typing import List
import logging
# import pickle


import torch
from nltk import sent_tokenize
from tqdm import tqdm, trange
# from transformers import AutoTokenizer
# from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from datasets import load_from_disk, Dataset, DatasetDict
import openai

# from dipper import DipperParaphraser
# from paraphrase_gen_utils import accept_by_bigram_overlap, SParrot
from paraphrase_gen_utils import gen_prompt  # query_openai, query_openai_bigram, gen_bigram_prompt, extract_list


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
    prompt = f"Previous context: {context} \nCurrent sentence to paraphrase: {sent} \nPlease provide {num_paraphrases} diverse paraphrased versions of this sentence in a numbered list."
    return prompt


def extract_list(text):
    """从 GPT 输出中提取列表项"""
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

    save_path = args.data_path + f'-0{start_i}:0{end_i}-para{num_paraphrases}'
    logger.info(f'save_path: {save_path}')
    fail_texts_index = []

    for i, text in enumerate(texts):  # tqdm(texts, desc="Paraphrasing with OpenAI"):
        sents = sent_tokenize(text)
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
            prompt = gen_prompt_v2(curr_sent, context, num_paraphrases=num_paraphrases)

            num_iter = 0
            para_ls = []

            while len(para_ls) < num_paraphrases and num_iter < max_loop:
                response = query_openai(client, prompt)
                para_ls = extract_list(response)
                logger.info(
                    f"getting [{i}][{j}] all paraphrased: {len(para_ls)} (should be {num_paraphrases} in total)"
                )
                num_iter += 1

            if len(para_ls) >= num_paraphrases:
                # para_sents.append(para_ls[:num_paraphrases])
                para_sents1.append(para_ls[0])
                para_sents2.append(para_ls[1])
                para_sents3.append(para_ls[2])
                para_sents4.append(para_ls[3])
                # para_sents5.append(para_ls[4])
                # para_sents6.append(para_ls[5])
                # para_sents7.append(para_ls[6])
                # para_sents8.append(para_ls[7])
            else:
                # 如果始终无法生成足够 paraphrase，跳过该句子
                is_all_done = False
                logger.warning(f"jump sentence: [{i}]")
                fail_texts_index.append(i)
                break

        if is_all_done:
            new_texts.append(sents)  # 合并成整段文本
            # all_paras.append(para_sents)  # 每个句子对应的 K 个 paraphrase 列表
            all_paras1.append(para_sents1)
            all_paras2.append(para_sents2)
            all_paras3.append(para_sents3)
            all_paras4.append(para_sents4)
            # all_paras5.append(para_sents5)
            # all_paras6.append(para_sents6)
            # all_paras7.append(para_sents7)
            # all_paras8.append(para_sents8)

    # 构建基础 Dataset
    dataset = Dataset.from_dict({
        'text': new_texts,
        'para_text1': all_paras1,
        'para_text2': all_paras2,
        'para_text3': all_paras3,
        'para_text4': all_paras4,
        # 'para_text5': all_paras5,
        # 'para_text6': all_paras6,
        # 'para_text7': all_paras7,
        # 'para_text8': all_paras8,
    })

    logger.info(f'dataset info: \n{dataset}')

    # 划分数据集：7:2:1 train:70%, validation:20%, test:10%
    train_test_split = dataset.train_test_split(test_size=0.3, seed=42)
    test_valid = train_test_split['test'].train_test_split(test_size=0.333)

    final_dataset = DatasetDict({
        'train': train_test_split['train'],
        'valid': test_valid['train'],
        'test': test_valid['test']
    })

    logger.info(f"final dataset info : {final_dataset}")

    # 保存到磁盘
    final_dataset.save_to_disk(save_path)
    logger.info(f"数据集已保存至：{save_path}")

    logger.info(f"释义失败的索引: {fail_texts_index}")

    return new_texts, 1


# python paraphrase_gen.py --data_path data/c4-train-8000
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
            'openai-bigram'
        ]
    )
    parser.add_argument('--temp', type=float, default=2.0, help='decode temperature')
    parser.add_argument(
        '--bert_threshold', type=float, default=0.0,
        help='threshold for bert similarity between original and paraphrased'
    )
    parser.add_argument('--num_beams', type=int, default=4, help='number of beams for beam-search')
    parser.add_argument('--max_iter', type=int, default=8, help='最大重新生成')
    args = parser.parse_args()
    logger.info(f"args:{args}")

    # tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = load_from_disk(args.data_path)
    logger.info(f"dataset info: {dataset}")
    start_i = 100
    end_i = 110
    logger.info(f"start_i: {start_i}, end_i: {end_i}")
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
    elif args.paraphraser == 'openai-bigram':
        client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL')
        )
        new_texts, paras = paraphrase_openai(client, texts, args.num_beams, bigram=True)
    else:
        raise NotImplementedError
