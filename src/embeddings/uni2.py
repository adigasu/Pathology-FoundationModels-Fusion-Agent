import torch
import timm
from typing import List, Any
from torchvision import transforms
from src.embeddings.base import BaseFoundationModelExtractor

class UNI2Extractor(BaseFoundationModelExtractor):
    """
    Feature extractor for UNI2-h (Mahmood Lab).
    ViT-H/14 trained on 100M+ histology patches.
    Default embedding dim: 1536.
    """
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_name="uni2", device=device)
        self.embed_dim = 1536
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def load_model(self):
        print(f"[UNI2] Loading MahmoodLab/UNI2-h on {self.device}...")
        timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667 * 2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
        self.model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        self.model.to(self.device).eval()
        self.model = self.model.half()  # Run in FP16 on GPU
        print(f"[UNI2] Successfully loaded UNI2-h (dim={self.embed_dim}) on {self.device}.")

    def get_embedding_dim(self) -> int:
        return self.embed_dim

    @torch.no_grad()
    def extract_batch(self, tile_images: List[Any]) -> torch.Tensor:
        tensors = torch.stack([self.transform(img) for img in tile_images]).to(self.device).half()
        features = self.model(tensors)
        return features.cpu()  # Retains FP16 on CPU
