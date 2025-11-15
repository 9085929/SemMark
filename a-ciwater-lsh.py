import torch
from datasets import load_from_disk
from dataclasses import dataclass
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from contrastive_trainer import ParaphraseDataset, SupConModel, SupConLoss
from embedder.openai_embedder import OpenaiEmbedder
from logger_config import setup_logger
from sampling_lsh_utils import get_mask_from_seed
from sbert_lsh_model import SBERTLSHModel, OpenAiLSHModel

LINE_COLOR = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
LINE_STYLE = ['-', '--', '-.', ':', '', '-', '-', ]
LINE_MARKER = ['o', 's', 'v', 'p', 'd', 'h', 'x', 'P', 'D', 'H', 'X', '^', '<', '>', '*']
LINE_WIDTH = 4

# 字体大小
TINY_SIZE = 4
SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 16
LARGE_SIZE = 20


def draw_boxes(
        y: list[list],
        y_axis_start: float = 0, y_axis_end: float = 1,
        x_axis_ticks: list = None, y_axis_ticks: list = None,
        x_axis_label: str = 'x', y_axis_label: str = 'y', title: str = 'fig',
        save_path: str = None, show: bool = True,
):
    fig, ax = plt.subplots()
    plt.grid(True)

    ax.boxplot(y, patch_artist=True, tick_labels=x_axis_ticks)

    ax.set_title(title, fontsize=BIGGER_SIZE)

    ax.tick_params(axis='x', labelsize=MEDIUM_SIZE)
    ax.set_xlabel(xlabel=x_axis_label, fontsize=MEDIUM_SIZE)

    ax.set_ylim(bottom=y_axis_start, top=y_axis_end)
    ax.set_yticks(ticks=y_axis_ticks)
    ax.tick_params(axis='y', labelsize=MEDIUM_SIZE, rotation=90)
    ax.set_ylabel(y_axis_label, fontsize=MEDIUM_SIZE)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path + '.png')
    else:
        plt.savefig(title)

    if show:
        plt.show()


def look():
    embedder = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sp_dim: int = 3  # Number of partitions in the embedding space. Default is 8.
    lmbd: float = 0.25  # Ratio of valid sentences.

    lsh_model = SBERTLSHModel(
        lsh_model_path=embedder, device=device, batch_size=1, lsh_dim=sp_dim, sbert_type='base'
    )

    text = "IF you like cricket, and like a brainy writer whose intellectualism is friendly and endearing, you\'d better read this book."
    lsh_seed = lsh_model.get_hash([text])[0]
    accept_mask = get_mask_from_seed(sp_dim, lmbd, lsh_seed)
    print(f"text={text}, lsh_seed={lsh_seed}, \naccept_mask={accept_mask} \n")

    text = "IF you like sport, and enjoy a brainy writer whose intellectualism is welcoming and charming, you\'d better read this book."
    lsh_seed_1 = lsh_model.get_hash([text])[0]
    accept_mask_1 = get_mask_from_seed(sp_dim, lmbd, lsh_seed_1)
    print(f"text={text},lsh_seed={lsh_seed_1}, \naccept_mask={accept_mask_1} \n")

    text = "IF you enjoy sport, and enjoy a brainy writer whose intellectualism is welcoming and charming, You should definitely check out this book."
    lsh_seed_2 = lsh_model.get_hash([text])[0]
    accept_mask_2 = get_mask_from_seed(sp_dim, lmbd, lsh_seed_2)
    print(f"text={text},lsh_seed={lsh_seed_2}, \naccept_mask={accept_mask_2} \n")

    text = "If you enjoy cricket and appreciate a clever writer whose intelligence is approachable and charming, you should definitely read this book."
    lsh_seed_3 = lsh_model.get_hash([text])[0]
    accept_mask_3 = get_mask_from_seed(sp_dim, lmbd, lsh_seed_3)
    print(f"text={text},lsh_seed={lsh_seed_3}, \naccept_mask={accept_mask_3} \n")

    text = "If you do not enjoy cricket and appreciate a clever writer whose intelligence is approachable and charming, you should definitely read this book."
    lsh_seed_3 = lsh_model.get_hash([text])[0]
    accept_mask_3 = get_mask_from_seed(sp_dim, lmbd, lsh_seed_3)
    print(f"text={text},lsh_seed={lsh_seed_3}, \naccept_mask={accept_mask_3} \n")

    text = "California is lifting its drought emergency for most of the state after a winter of record rain and snowfall that followed a five-year dry spell."
    lsh_seed = lsh_model.get_hash([text])[0]
    accept_mask = get_mask_from_seed(sp_dim, lmbd, lsh_seed)
    print(f"text={text}, lsh_seed={lsh_seed}, \naccept_mask={accept_mask} \n")

    text = "California is ending its drought emergency across much of the state following a winter of unprecedented precipitation and snow accumulation that came after five years of arid conditions."
    lsh_seed = lsh_model.get_hash([text])[0]
    accept_mask = get_mask_from_seed(sp_dim, lmbd, lsh_seed)
    print(f"text={text}, lsh_seed={lsh_seed}, \naccept_mask={accept_mask} \n")

    accept_mask = get_mask_from_seed(sp_dim, lmbd, lsh_seed + 1)
    print(f"lsh_seed={lsh_seed + 1}, \naccept_mask={accept_mask} \n")

    accept_mask = get_mask_from_seed(sp_dim, lmbd, lsh_seed - 2)
    print(f"lsh_seed={lsh_seed - 2}, \naccept_mask={accept_mask}")


def main_v1():
    loaded_dataset = load_from_disk('dataset/c4/80train20test-8000-0000:0100-para8')
    logger.info(loaded_dataset)
    test_dataset = loaded_dataset['test']
    eval_dataset = loaded_dataset['valid']
    train_dataset = loaded_dataset['train']
    texts = test_dataset['text'] + eval_dataset['text'] + train_dataset['text']
    para1 = test_dataset['para_text1'] + eval_dataset['para_text1'] + train_dataset['para_text1']
    para2 = test_dataset['para_text2'] + eval_dataset['para_text2'] + train_dataset['para_text2']
    para3 = test_dataset['para_text3'] + eval_dataset['para_text3'] + train_dataset['para_text3']
    para4 = test_dataset['para_text4'] + eval_dataset['para_text4'] + train_dataset['para_text4']
    para5 = test_dataset['para_text5'] + eval_dataset['para_text5'] + train_dataset['para_text5']
    para6 = test_dataset['para_text6'] + eval_dataset['para_text6'] + train_dataset['para_text6']
    para7 = test_dataset['para_text7'] + eval_dataset['para_text7'] + train_dataset['para_text7']
    para8 = test_dataset['para_text8'] + eval_dataset['para_text8'] + train_dataset['para_text8']

    # "/home/haojifei/dev_resource/huggingface/models/AbeHou/SemStamp-c4-sbert"
    # "/home/haojifei/dev_resource/huggingface/models/sentence-transformers/all-mpnet-base-v1"
    # "/home/haojifei/dev_resource/huggingface/models/Qwen/Qwen3-Embedding-8B"
    # "/home/haojifei/dev_resource/huggingface/models/Qwen/Qwen3-Embedding-4B"
    # "/home/haojifei/dev_resource/huggingface/models/Qwen/Qwen3-Embedding-0.6B"
    embedder_path = "/home/haojifei/dev_resource/huggingface/models/Qwen/Qwen3-Embedding-0.6B"
    logger.info(f"Loading embedder from {embedder_path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lsh_model = SBERTLSHModel(
        device=device, batch_size=1, lsh_dim=3,
        sbert_type='base', lsh_model_path=embedder_path
    )

    # embedder = OpenaiEmbedder(
    #     model_name='text-embedding-3-large',
    #     model_api_key='sk-d9GXiBuj1oYhyrqF4963E03d62Da4cF3A646E648E509A700',
    #     model_api_base_url='https://chatapi.onechats.ai/v1/',  # "https://chatapi.onechats.top/v1/"
    # )
    # lsh_model = OpenAiLSHModel('cpu', 1, 3, embedder_path, 768)

    lsh_success: list[float] = []
    for t, p1, p2, p3, p4, p5, p6, p7, p8 in zip(texts, para1, para2, para3, para4, para5, para6, para7, para8):
        sentence_t = ' '.join(t)
        sentence_p1 = ' '.join(p1)
        sentence_p2 = ' '.join(p2)
        sentence_p3 = ' '.join(p3)
        sentence_p4 = ' '.join(p4)
        sentence_p5 = ' '.join(p5)
        sentence_p6 = ' '.join(p6)
        sentence_p7 = ' '.join(p7)
        sentence_p8 = ' '.join(p8)
        lsh_seed = lsh_model.get_hash(
            [sentence_t, sentence_p1, sentence_p2, sentence_p3, sentence_p4,
             sentence_p5, sentence_p6, sentence_p7, sentence_p8]
        )
        count: int = 0
        for i in range(1, len(lsh_seed)):
            if lsh_seed[i] == lsh_seed[0]:
                count += 1
        lsh_success.append(count / (len(lsh_seed) - 1))
        logger.info(count)
    logger.info(lsh_success)

    embedder_path_dirs = embedder_path.split('/')
    tag = embedder_path_dirs[-1]
    draw_boxes(
        [lsh_success],
        y_axis_start=0, y_axis_end=1,
        x_axis_ticks=[tag], y_axis_ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
        x_axis_label='M', y_axis_label='success rate',
        title=tag + "-ContrastiveTrained", save_path=f'{tag}.png'
    )
    logger.info(f"Save results to {tag}.png")


def train(cfg):
    dataset = ParaphraseDataset(cfg.train_path, cfg)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = SupConModel(cfg).to(cfg.device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = SupConLoss(temperature=cfg.temperature)

    total_steps = len(loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    for epoch in range(cfg.epochs):
        model.train()
        losses = []

        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}"):
            anchor_texts, pos_texts, neg_texts = batch
            texts = []
            labels = []

            for a, p, n in zip(anchor_texts, pos_texts, neg_texts):
                texts.append(a)
                labels.append(0)
                texts.append(p)
                labels.append(0)
                texts.append(n)
                labels.append(1)

            labels = torch.tensor(labels, dtype=torch.long).to(cfg.device)

            features = model(texts)  # [3B, 768]
            loss = loss_fn(features, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

        print(f"Epoch {epoch + 1}: loss={sum(losses) / len(losses):.4f}")

    torch.save(model.encoder.state_dict(), "encoder_finetuned.pt")
    print("Saved encoder_finetuned.pt")


@dataclass
class Config:
    train_path: str = "data.json"
    model_name: str = "/home/haojifei/dev_resource/huggingface/models/FacebookAI/roberta-base"
    lr: float = 2e-5
    epochs: int = 10
    batch_size: int = 16
    max_len: int = 128
    temperature: float = 0.07
    device: str = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == '__main__':
    logger = setup_logger(__name__, log_file=f'{__file__}.log')
    logger.info("=" * 30)
    # look()
    # main_v1()
    cfg = Config()
    train(cfg)
