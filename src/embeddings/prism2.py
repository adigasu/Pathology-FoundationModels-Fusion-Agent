import torch
from typing import List, Any
from torchvision import transforms
from transformers import AutoModel
from src.embeddings.base import BaseFoundationModelExtractor

class Prism2Extractor(BaseFoundationModelExtractor):
    """
    Feature extractor for Prism2 (Paige AI).
    Multi-modal foundation model loaded via Hugging Face Transformers.
    """
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_name="prism2", device=device)
        self.embed_dim = 1280
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def load_model(self):
        print(f"[Prism2] Loading paige-ai/Prism2 on {self.device}...")
        self.model = AutoModel.from_pretrained("paige-ai/Prism2", trust_remote_code=True)
        self.model.to(self.device).eval()
        self.model = self.model.half()
        print(f"[Prism2] Successfully loaded Prism2 on {self.device}.")

    def get_embedding_dim(self) -> int:
        return self.embed_dim

    @torch.no_grad()
    def extract_batch(self, tile_images: List[Any]) -> torch.Tensor:
        tensors = torch.stack([self.transform(img) for img in tile_images]).to(self.device).half()
        output = self.model(tensors)
        if hasattr(output, 'last_hidden_state'):
            features = output.last_hidden_state[:, 0]
        elif hasattr(output, 'pooler_output') and output.pooler_output is not None:
            features = output.pooler_output
        elif isinstance(output, tuple):
            features = output[0]
        else:
            features = output
        return features.cpu()
