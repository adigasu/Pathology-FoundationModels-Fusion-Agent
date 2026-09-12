import torch
import timm
from typing import List, Any
from torchvision import transforms
from src.embeddings.base import BaseFoundationModelExtractor

class GigaPathExtractor(BaseFoundationModelExtractor):
    """
    Feature extractor for Prov-GigaPath (Microsoft / Providence).
    ViT-Giant/14 architecture pretrained on 1.3B pathology image patches.
    Default embedding dimension: 1536.
    """
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_name="gigapath", device=device)
        self.embed_dim = 1536
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def load_model(self):
        print(f"[GigaPath] Loading prov-gigapath/prov-gigapath on {self.device}...")
        self.model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        self.model.to(self.device).eval()
        self.model = self.model.half()  # Run in FP16 on GPU
        print(f"[GigaPath] Successfully loaded Prov-GigaPath (dim={self.embed_dim}) on {self.device}.")

    def get_embedding_dim(self) -> int:
        return self.embed_dim

    @torch.no_grad()
    def extract_batch(self, tile_images: List[Any]) -> torch.Tensor:
        tensors = torch.stack([self.transform(img) for img in tile_images]).to(self.device).half()
        features = self.model(tensors)  # Output shape: [B, 1536]
        return features.cpu()  # Retain FP16 on CPU
