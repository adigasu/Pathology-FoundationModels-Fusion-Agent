import torch
import torch.nn as nn
import torch.nn.functional as F

class SlideAggregator(nn.Module):
    """
    Aggregates variable-length tile embeddings [N_tiles, D] into slide-level representation [D].
    Supports:
    - 'mean': Mean pooling
    - 'max': Max pooling
    - 'abmil': Attention-Based Multiple Instance Learning (Ilse et al., 2018)
    """
    def __init__(self, method: str = "mean", embed_dim: int = 1024, hidden_dim: int = 256):
        super().__init__()
        self.method = method.lower()
        self.embed_dim = embed_dim

        if self.method == "abmil":
            self.attention_V = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.Tanh()
            )
            self.attention_U = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.Sigmoid()
            )
            self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [N, D] or [B, N, D]
        Returns:
            Tensor of shape [D] or [B, D]
        """
        if x.dim() == 2:
            # Single slide: [N, D]
            if self.method == "mean":
                return torch.mean(x, dim=0)
            elif self.method == "max":
                return torch.max(x, dim=0)[0]
            elif self.method == "abmil":
                # Gated attention mechanism
                a_V = self.attention_V(x)  # [N, hidden]
                a_U = self.attention_U(x)  # [N, hidden]
                a = self.attention_weights(a_V * a_U)  # [N, 1]
                a = torch.softmax(a, dim=0)  # [N, 1]
                slide_embed = torch.sum(a * x, dim=0)  # [D]
                return slide_embed
            else:
                raise ValueError(f"Unknown pooling method: {self.method}")
        elif x.dim() == 3:
            # Batched slides: [B, N, D]
            if self.method == "mean":
                return torch.mean(x, dim=1)
            elif self.method == "max":
                return torch.max(x, dim=1)[0]
            elif self.method == "abmil":
                a_V = self.attention_V(x)
                a_U = self.attention_U(x)
                a = self.attention_weights(a_V * a_U)
                a = torch.softmax(a, dim=1)
                return torch.sum(a * x, dim=1)
            else:
                raise ValueError(f"Unknown pooling method: {self.method}")
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")

    def aggregate(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
