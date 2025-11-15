from typing import List, Tuple, Callable, Optional, Iterator
import os
from copy import deepcopy

import numpy as np
from nearpy.hashes import RandomBinaryProjections
from scipy.spatial.distance import hamming, cosine
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
# from sim_api import get_vec_para_repl
import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from embedder.openai_embedder import OpenaiEmbedder

global Device


def batched(iterable, n, total=None):
    l = len(iterable)
    if total is not None:
        assert l == total
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


class LSHModel:
    def __init__(self, device: str, batch_size: int, lsh_dim: int = 3):
        self.device = device
        self.batch_size: int = batch_size
        self.lsh_dim: int = lsh_dim

        # self.comparator = lambda x, y: hamming(*[
        #     np.fromstring(self.hasher.hash_vector(i)[0], 'u1') - ord('0') for i in [x, y]
        # ])
        # self.comparator = lambda x, y: cosine(x, y)
        self.comparator: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: cosine(x, y)
        self.hasher = RandomBinaryProjections(
            'rbp_perm', projection_count=self.lsh_dim, rand_seed=2049
        )
        self.do_lsh: bool = True
        self.dimension: int = -1

    def compute_distances(self, refs: List[str], candidates: List[str]) -> np.ndarray:
        """
        :param refs: list of reference sentences
        :param candidates: list of candidate sentences to compute similarity distances from references
        :return:
        """
        assert len(refs) == len(candidates), f"refs size {len(refs)} != candidates size {len(candidates)}"
        results = np.zeros(len(refs))
        i = 0
        for batch in batched(zip(refs, candidates), self.batch_size, total=len(refs)):
            (ref_b, cands_b) = list(zip(*batch))
            assert len(ref_b) <= self.batch_size
            [ref_features, cand_features] = [self.get_embeddings(x) for x in [ref_b, cands_b]]

            if i == 0:
                print(f"comparing vectors of dimension {ref_features.shape[-1]}")

            results[i:i + len(ref_b)] = np.fromiter(
                map(lambda args: self.comparator(*args), zip(ref_features, cand_features)), dtype=float)
            i += len(ref_b)

        return results

    def get_embeddings(self, sents: Iterator[str]) -> np.ndarray:
        """
        retrieve np array of sentence embeddings from sentence iterator
        :param sents: set of sentence strings
        :return: extracted embeddings
        """
        raise NotImplementedError()

    def get_hash(self, sents: Iterator[str] | List[str]) -> List[int]:  # todo: Iterator[str]:
        embd = self.get_embeddings(sents)
        # print(f"embedding: {embd}")
        hash_strs = [self.hasher.hash_vector(e)[0] for e in embd]
        hash_ints = [int(s, 2) for s in hash_strs]
        return hash_ints


class SBERTLSHModel(LSHModel):

    def __init__(
            self, device: str, batch_size: int, lsh_dim: int = 3,
            sbert_type: str = 'roberta', lsh_model_path: str = None,
            **kwargs
    ):
        """
        :param lsh_dim: Number of partitions in the embedding space. Default is 8.
        :param lsh_model_path:
        """
        super(SBERTLSHModel, self).__init__(device, batch_size, lsh_dim)
        if lsh_model_path is None:
            # "/home/haojifei/dev_resource/huggingface/models/Qwen/Qwen3-Embedding-8B"
            lsh_model_path = "/home/haojifei/dev_resource/huggingface/models/Qwen/Qwen3-Embedding-8B"

        self.sbert_type = sbert_type
        # self.dimension = 1024 if 'large' in self.sbert_type else 768

        self.embedder = SentenceTransformer(lsh_model_path, device=self.device)
        self.dimension = self.embedder.get_sentence_embedding_dimension()
        # self.embedder.eval()

        self.hasher.reset(dim=self.dimension)

    def get_embeddings(self, sents: Iterator[str] | List[str]) -> np.ndarray:
        all_embeddings = self.embedder.encode(sents, batch_size=len(sents))
        return np.stack(all_embeddings)


class OpenAiLSHModel(LSHModel):
    def __init__(
            self, device, batch_size, lsh_dim,
            openai_embedder: OpenaiEmbedder,
            openai_embedder_dim: int = 786
    ):
        super(OpenAiLSHModel, self).__init__(device, batch_size, lsh_dim)
        self.embedder = openai_embedder
        self.dimension = openai_embedder_dim
        self.hasher.reset(dim=self.dimension)

    def get_embeddings(self, sents: Iterator[str] | List[str]) -> np.ndarray:
        embedding = self.embedder.get_embeddings(sents, self.dimension)
        embedding_np = np.array(embedding, dtype=np.float32)
        return np.stack(embedding_np)
