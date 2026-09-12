import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
import argparse
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any

from src.embeddings.cache import FeatureCache
from src.embeddings.aggregator import SlideAggregator
from src.embeddings.uni2 import UNI2Extractor
from src.embeddings.virchow2 import Virchow2Extractor
from src.embeddings.prism2 import Prism2Extractor
from src.embeddings.gigapath import GigaPathExtractor

Image.MAX_IMAGE_PIXELS = None

EXTRACTORS = {
    'uni2': UNI2Extractor,
    'virchow2': Virchow2Extractor,
    'gigapath': GigaPathExtractor,
    'prism2': Prism2Extractor,
}

def resolve_data_dir(cli_dir: str = None) -> str:
    if cli_dir and os.path.exists(cli_dir):
        return cli_dir
    candidates = [
        os.environ.get("LUNG_DATA_DIR", ""),
        os.path.expanduser("~/lung_data"),
        "./data/lung_data",
        # Optional custom user path configured via LUNG_DATA_DIR
        os.environ.get("CUSTOM_LUNG_DIR", "")
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return candidates[1]

def extract_slide_tiles(wsi_path: str, tile_meta: Dict[str, Any], extractor, batch_size: int = 64) -> torch.Tensor:
    """
    Extracts all tile images from a slide and passes them through the extractor in batches.
    Returns:
        torch.Tensor of shape [N_tiles, D] in float16 precision.
    """
    with Image.open(wsi_path) as img:
        w_orig, h_orig = img.size
        # Fast draft decode at 20x equivalent (1/4 scale)
        w_20x, h_20x = w_orig // 4, h_orig // 4
        img.draft('RGB', (w_20x, h_20x))
        img_20x = img.resize((w_20x, h_20x), Image.Resampling.BILINEAR)

    tiles_info = tile_meta['tiles']
    num_tiles = len(tiles_info)
    tile_size = 224

    all_embeddings = []

    for b_start in range(0, num_tiles, batch_size):
        b_end = min(b_start + batch_size, num_tiles)
        batch_tiles = tiles_info[b_start:b_end]

        batch_crops = []
        for t in batch_tiles:
            x_20x, y_20x = t['x_20x'], t['y_20x']
            crop = img_20x.crop((x_20x, y_20x, x_20x + tile_size, y_20x + tile_size))
            batch_crops.append(crop)

        with torch.no_grad():
            embeds = extractor.extract_batch(batch_crops)  # shape: [B, D]
            all_embeddings.append(embeds.half().cpu())

    if all_embeddings:
        return torch.cat(all_embeddings, dim=0)
    else:
        return torch.empty((0, extractor.embedding_dim), dtype=torch.float16)

def main():
    parser = argparse.ArgumentParser(description="Step 1.5: Foundation Model Feature Extraction (FP16 Tile Caching)")
    parser.add_argument("--model", type=str, required=True, choices=['uni2', 'virchow2', 'gigapath', 'prism2'])
    parser.add_argument("--img_dir", type=str, default=None, help="Directory containing raw WSI JPGs")
    parser.add_argument("--tiles_json", type=str, default="artifacts/tiles_metadata.json")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_slides", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.img_dir = resolve_data_dir(args.img_dir)

    print("=" * 75)
    print(f"STEP 1.5: FEATURE EXTRACTION — MODEL: {args.model.upper()}")
    print(f"Image Directory : {args.img_dir}")
    print(f"Device: {args.device} | Batch Size: {args.batch_size} | Tile Precision: FP16")
    print("=" * 75)

    if not os.path.exists(args.tiles_json):
        print(f"ERROR: Tiles metadata {args.tiles_json} not found. Run scripts/03_tile_wsi.py first.")
        sys.exit(1)

    with open(args.tiles_json, 'r', encoding='utf-8') as f:
        tiles_metadata = json.load(f)

    extractor_cls = EXTRACTORS[args.model]
    extractor = extractor_cls(device=args.device)
    extractor.load_model()

    cache = FeatureCache(cache_dir="artifacts/features")
    mean_aggregator = SlideAggregator(method="mean")
    max_aggregator = SlideAggregator(method="max")

    slide_ids = list(tiles_metadata.keys())
    if args.max_slides:
        slide_ids = slide_ids[:args.max_slides]
        print(f"Processing limited cohort of {len(slide_ids)} slides.")

    tile_cache_dict = {}
    slide_mean_dict = {}
    slide_max_dict = {}

    start_time = time.time()
    for idx, wsi_id in enumerate(slide_ids, 1):
        meta = tiles_metadata[wsi_id]
        wsi_path = os.path.join(args.img_dir, f"{wsi_id}.jpg")

        if not os.path.exists(wsi_path):
            print(f"[{idx:3d}/{len(slide_ids)}] {wsi_id}: Skipped (Image missing on disk)")
            continue

        t0 = time.time()
        tile_embeds = extract_slide_tiles(wsi_path, meta, extractor, batch_size=args.batch_size)
        dt = time.time() - t0

        # Store tile-level embeddings in FP16
        tile_cache_dict[wsi_id] = {
            'patient_id': meta['patient_id'],
            'subtype': meta['subtype'],
            'age': meta['age'],
            'sex': meta['sex'],
            'n_tiles': tile_embeds.shape[0],
            'embeddings': tile_embeds  # torch.HalfTensor [N, D]
        }

        # Compute slide-level representations (FP32)
        if tile_embeds.shape[0] > 0:
            slide_mean = mean_aggregator.aggregate(tile_embeds.float())
            slide_max = max_aggregator.aggregate(tile_embeds.float())
            slide_mean_dict[wsi_id] = slide_mean
            slide_max_dict[wsi_id] = slide_max

        print(f"[{idx:3d}/{len(slide_ids)}] {wsi_id} -> {tile_embeds.shape[0]} tiles in {dt:.2f}s ({tile_embeds.shape[0]/max(0.01, dt):.1f} tiles/s)")

    total_dt = time.time() - start_time
    print("-" * 75)
    print(f"Extracted features for {len(tile_cache_dict)} slides in {total_dt/60:.2f} minutes.")

    # 1. Save tile-level FP16 embeddings
    cache.save_tile_features(args.model, tile_cache_dict)

    # 2. Save slide-level aggregated vectors
    cache.save_slide_features(args.model, "mean", slide_mean_dict)
    cache.save_slide_features(args.model, "max", slide_max_dict)
    print(f"Feature caching for {args.model.upper()} completed successfully!")

if __name__ == "__main__":
    main()
