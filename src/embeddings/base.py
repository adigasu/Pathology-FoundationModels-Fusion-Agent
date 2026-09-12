import abc
import torch
from typing import List, Dict, Any

class BaseFoundationModelExtractor(abc.ABC):
    """
    Abstract base class for pathology foundation model feature extraction.
    Ensures consistent API, per-model transforms, and FP16 tile embedding outputs.
    """
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.transform = None

    @abc.abstractmethod
    def load_model(self):
        """Loads model weights and initializes evaluation mode."""
        pass

    @abc.abstractmethod
    def get_embedding_dim(self) -> int:
        """Returns the embedding vector dimension."""
        pass

    @abc.abstractmethod
    def extract_batch(self, tile_images: List[Any]) -> torch.Tensor:
        """
        Extracts patch embeddings for a batch of PIL Images.
        Returns:
            torch.Tensor of shape [B, D] in float16 precision.
        """
        pass
