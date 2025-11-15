from typing import List

import openai
import numpy as np

from embedder.embedder import Embedder


class OpenaiEmbedder(Embedder):
    def __init__(self, model_name: str, model_api_key: str, model_api_base_url: str = None):
        self.model_api_key = model_api_key
        self.model_api_base_url = model_api_base_url
        super().__init__(model_name)

    def _load_model(self):
        if self.model_api_base_url:
            client = openai.OpenAI(
                api_key=self.model_api_key,
                base_url=self.model_api_base_url,
            )
        else:
            client = openai.OpenAI(api_key=self.model_api_key)
        return client

    def get_embeddings(self, words: List[str], embedding_dimension: int = 256) -> list[list[float]]:
        response = self.model.embeddings.create(
            model=self.model_name,
            input=words,
            encoding_format="float",
            dimensions=embedding_dimension
        )
        embeddings = [r.embedding for r in response.data]
        # array = np.array(embeddings, dtype=np.float64)
        return embeddings

    def get_embedding(self, text: str, embedding_dimension: int = 256) -> list[float]:
        response = self.model.embeddings.create(
            model=self.model_name,
            input=text,
            encoding_format="float",
            dimensions=embedding_dimension
        )
        embedding = response.data[0].embedding
        return embedding
