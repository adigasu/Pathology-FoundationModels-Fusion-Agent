import torch
import timm
from timm.layers import SwiGLUPacked
from typing import List, Any
from torchvision import transforms
from src.embeddings.base import BaseFoundationModelExtractor

class Virchow2Extractor(BaseFoundationModelExtractor):
    """
    Feature extractor for Virchow2 (Paige AI).
    ViT-H/14 architecture with SwiGLU activations.
    Official representation: Class token concatenated with average patch tokens (dim=2560)
    or class token alone (dim=1280).
    """
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu", use_2560: bool = True):
        super().__init__(model_name="virchow2", device=device)
        self.use_2560 = use_2560
        self.embed_dim = 2560 if use_2560 else 1280
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def load_model(self):
        print(f"[Virchow2] Loading paige-ai/Virchow2 on {self.device}...")
        self.model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU
        )
        self.model.to(self.device).eval()
        self.model = self.model.half()  # Run in FP16 on GPU
        print(f"[Virchow2] Successfully loaded Virchow2 (dim={self.embed_dim}) on {self.device}.")

    def get_embedding_dim(self) -> int:
        return self.embed_dim

    @torch.no_grad()
    def extract_batch(self, tile_images: List[Any]) -> torch.Tensor:
        tensors = torch.stack([self.transform(img) for img in tile_images]).to(self.device).half()
        output = self.model(tensors)  # [B, 261, 1280]

        if self.use_2560:
            class_token = output[:, 0]       # [B, 1280]
            patch_tokens = output[:, 5:]     # [B, 256, 1280] (tokens 1-4 are register tokens)
            features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # [B, 2560]
        else:
            features = output[:, 0]          # [B, 1280]

        return features.cpu()  # Retains FP16 on CPU
