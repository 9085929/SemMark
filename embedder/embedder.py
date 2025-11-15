from abc import ABC, abstractmethod


class Embedder(ABC):
    """
    The EMBEDDER needs to be capable of projecting both words and documents into a high-dimensional latent space.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self._load_model()

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def get_embeddings(self, words: list[str], embedding_dimension: int = 256) -> list[list[float]]:
        pass

    @abstractmethod
    def get_embedding(self, text: str, embedding_dimension: int = 256) -> list[float]:
        """
        Args:
            text: could be a word, a sentence, a paragraph or a document.
            embedding_dimension:
        Returns:
            A high-dimensional vector of the input text.
        """
        pass
