from collections import Counter
import openai
import torch
import re
from tqdm import trange

device = 'cuda' if torch.cuda.is_available() else "cpu"
from parrot import Parrot

stops = []

class SParrot(Parrot):
    def __init__(self, model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False):
        super().__init__(model_tag, use_gpu)

    def augment(self, input_phrase, use_gpu=False, diversity_ranker="levenshtein", do_diverse=False,
                max_return_phrases=10, max_length=32, adequacy_threshold=0.90, fluency_threshold=0.90):
        if use_gpu:
            device = "cuda:0"
        else:
            device = "cpu"

        self.model = self.model.to(device)

        import re

        save_phrase = input_phrase
        if len(input_phrase) >= max_length:
            max_length += 32

        input_phrase = re.sub('[^a-zA-Z0-9 \?\'\-\/\:\.]', '', input_phrase)
        input_phrase = "paraphrase: " + input_phrase
        input_ids = self.tokenizer.encode(input_phrase, return_tensors='pt')
        input_ids = input_ids.to(device)

        if do_diverse:
            for n in range(2, 9):
                if max_return_phrases % n == 0:
                    break
                    # print("max_return_phrases - ", max_return_phrases , " and beam groups -", n)
            preds = self.model.generate(
                input_ids,
                do_sample=False,
                max_length=max_length,
                num_beams=max_return_phrases,
                num_beam_groups=n,
                diversity_penalty=2.0,
                early_stopping=True,
                num_return_sequences=max_return_phrases)
        else:
            preds = self.model.generate(
                input_ids,
                do_sample=True,
                max_length=max_length,
                top_k=50,
                top_p=0.95,
                early_stopping=True,
                num_return_sequences=max_return_phrases)

        paraphrases = set()

        for pred in preds:
            gen_pp = self.tokenizer.decode(pred, skip_special_tokens=True).lower()
            gen_pp = re.sub('[^a-zA-Z0-9 \?\'\-]', '', gen_pp)
            paraphrases.add(gen_pp)

        adequacy_filtered_phrases = self.adequacy_score.filter(input_phrase, paraphrases, adequacy_threshold, device)
        if len(adequacy_filtered_phrases) == 0:
            adequacy_filtered_phrases = paraphrases
        fluency_filtered_phrases = self.fluency_score.filter(adequacy_filtered_phrases, fluency_threshold, device)
        if len(fluency_filtered_phrases) == 0:
            fluency_filtered_phrases = adequacy_filtered_phrases
        diversity_scored_phrases = self.diversity_score.rank(input_phrase, fluency_filtered_phrases, diversity_ranker)
        para_phrases = []
        for para_phrase, diversity_score in diversity_scored_phrases.items():
            para_phrases.append((para_phrase, diversity_score))
        para_phrases.sort(key=lambda x: x[1], reverse=True)
        para_phrases = [x[0] for x in para_phrases]
        return para_phrases


def tokenize(tokenizer, text):
    return tokenizer(text, return_tensors='pt').input_ids[0].to(device)


def build_bigrams(input_ids):
    bigrams = []
    for i in range(len(input_ids) - 1):
        bigram = tuple(input_ids[i:i + 2].tolist())
        bigrams.append(bigram)
    return bigrams


def extract_list(text):
    p = re.compile("^[0-9]+[.)\]\*·:] (.*(?:\n(?![0-9]+[.)\]\*·:]).*)*)", re.MULTILINE)
    return p.findall(text)


def compare_ngram_overlap(input_ngram, para_ngram):
    input_c = Counter(input_ngram)
    para_c = Counter(para_ngram)
    intersection = list(input_c.keys() & para_c.keys())
    overlap = 0
    for i in intersection:
        overlap += para_c[i]
    return overlap

def gen_prompt(sent, context):
    prompt = f'''Previous context: {context} \n Current sentence to paraphrase: {sent}'''
    return prompt


def gen_bigram_prompt(sent, context, num_beams):
    prompt = f'''Previous context: {context} \n Paraphrase in {num_beams} different ways and return a numbered list : {sent}'''
    return prompt
def gen_prompt_synonym(sent, context, num_paraphrases=5):
    prompt = (
    f"Previous context: {context}\n"
    f"Current sentence to paraphrase: {sent}\n"
    f"Please provide {num_paraphrases} rewritten versions in a numbered list.\n"
    f"Requirements for Strict Synonym Substitution Attack:\n"
    f"1. Replace content words (verbs, adjectives, adverbs, and general nouns) with contextually appropriate synonyms whenever possible.\n"
    f"2. Perform substitutions primarily through direct lexical replacement rather than phrase rewriting, while preserving fluency.\n"
    f"3. Do NOT modify named entities, technical terms, numbers, or key subject nouns.\n"
    f"4. Preserve the original semantic meaning, factual content, sentiment, and logical flow. Do not add or remove information.\n"
    f"5. Keep sentence structure, clause order, and voice unchanged except for minimal grammatical adjustments required for natural language.\n"
)
    return prompt
def gen_prompt_pure_bt_step1(sent):
    prompt = (
        f"Translate the following sentence into French. Preserve the meaning exactly. Output ONLY the French translation.\n"
        f"Sentence: {sent}"
    )
    return prompt

def gen_prompt_pure_bt_step2(french_sent):
    prompt = (
        f"Translate the following French sentence back into English. The result should be fluent and semantically equivalent to the original. Output ONLY the English translation.\n"
        f"Sentence: {french_sent}"
    )
    return prompt
def gen_prompt_enhanced_paraphrase(sent, context, num_paraphrases=4):
    prompt = (
        f"Previous context: {context}\n"
        f"Current sentence: {sent}\n"
        f"Please perform a standard translation-backtranslation process using French as the intermediate language.\n"
        f"Instructions:\n"
        f"1. Translate the current sentence into French.\n"
        f"2. Translate that French version back into English.\n"
        f"3. Ensure the final English sentence preserves the original meaning, facts, and entities.\n"
        f"4. Allow natural changes in wording and sentence structure resulting from the translation process.\n"
        f"5. Output ONLY {num_paraphrases} final back-translated English versions in a numbered list. Do NOT output the intermediate French text.\n"
    )
    return prompt
def gen_prompt_paper_regular(sent, context):
    prompt = (
        f"Previous context: {context}\n"
        f"Current sentence to paraphrase: {sent}"
    )
    return prompt

def query_openai(client, prompt, temperature=1.0): 
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature, # 使用传入的温度
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API 调用出错: {e}")
        return "" 

