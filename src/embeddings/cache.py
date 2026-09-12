import os
import torch
import numpy as np
from typing import Dict, Any, Optional

class FeatureCache:
    """
    Manages loading and saving of FP16 tile-level embeddings and slide-level features,
    with native memory-mapping (mmap=True) support for instant lazy-loading.
    """
    def __init__(self, cache_dir: str = "artifacts/features"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_tile_cache_path(self, model_name: str) -> str:
        return os.path.join(self.cache_dir, f"{model_name}_tiles_fp16.pt")

    def get_slide_cache_path(self, model_name: str, method: str = "mean") -> str:
        return os.path.join(self.cache_dir, f"{model_name}_slide_{method}.pt")

    def save_tile_features(self, model_name: str, features_dict: Dict[str, Any]):
        """
        Saves tile-level embeddings dictionary:
        {
           wsi_id: {
               'patient_id': str,
               'subtype': str,
               'age': int,
               'sex': str,
               'n_tiles': int,
               'embeddings': torch.HalfTensor [N, D]  # FP16
           }
        }
        """
        path = self.get_tile_cache_path(model_name)
        torch.save(features_dict, path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"[Cache] Saved {len(features_dict)} slide tile embeddings in FP16 to: {path} ({size_mb:.1f} MB)")

    def load_tile_features(self, model_name: str, mmap: bool = True) -> Dict[str, Any]:
        """
        Loads tile-level embeddings with optional memory-mapping (mmap=True).
        """
        path = self.get_tile_cache_path(model_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tile features cache not found at: {path}")
        try:
            return torch.load(path, map_location="cpu", mmap=mmap)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def save_slide_features(self, model_name: str, method: str, slide_dict: Dict[str, Any]):
        path = self.get_slide_cache_path(model_name, method)
        torch.save(slide_dict, path)
        print(f"[Cache] Saved aggregated slide embeddings ({method}) to: {path}")

    def load_slide_features(self, model_name: str, method: str = "mean") -> Dict[str, Any]:
        path = self.get_slide_cache_path(model_name, method)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Slide features cache not found at: {path}")
        return torch.load(path, map_location="cpu")
