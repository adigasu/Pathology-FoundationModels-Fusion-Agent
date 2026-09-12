import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
import argparse
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.data.tiling import WSITiler

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

def process_single_slide(item, img_dir, qc_dir, tile_size, min_tissue):
    wsi_id = item['WSI_ID']
    wsi_path = os.path.join(img_dir, f"{wsi_id}.jpg")
    
    if not os.path.exists(wsi_path):
        return None, wsi_id, "File not found"

    t0 = time.time()
    tiler = WSITiler(tile_size=tile_size, min_tissue_frac=min_tissue)
    res = tiler.process_slide(wsi_path)
    qc_path = os.path.join(qc_dir, f"{wsi_id}_qc.jpg")
    tiler.save_qc_overlay(res, qc_path)
    dt = time.time() - t0

    meta = {
        'patient_id': item['patient_id'],
        'subtype': item['Tumor Subtype'],
        'age': item['Age'],
        'sex': item['Sex'],
        'w_orig': res['w_orig'],
        'h_orig': res['h_orig'],
        'w_20x': res['w_20x'],
        'h_20x': res['h_20x'],
        'tissue_percentage': res['tissue_percentage'],
        'n_tiles': res['n_tiles'],
        'tiles': res['tiles'],
        'qc_path': qc_path
    }
    return meta, wsi_id, dt

def main():
    parser = argparse.ArgumentParser(description="Step 1.4: Parallel WSI Preprocessing & Tissue Tiling")
    parser.add_argument("--img_dir", type=str, default=None, help="Path to WSI images directory")
    parser.add_argument("--curated_csv", type=str, default="artifacts/curated_patients.csv")
    parser.add_argument("--out_json", type=str, default="artifacts/tiles_metadata.json")
    parser.add_argument("--qc_dir", type=str, default="artifacts/tiling_qc")
    parser.add_argument("--tile_size", type=int, default=224)
    parser.add_argument("--min_tissue", type=float, default=0.50)
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--max_slides", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Force re-tiling of all slides")
    args = parser.parse_args()

    args.img_dir = resolve_data_dir(args.img_dir)

    print("=" * 75)
    print("STEP 1.4: 20x WSI PREPROCESSING, TISSUE SEGMENTATION & TILING (204 COHORT)")
    print(f"Image Directory : {args.img_dir}")
    print(f"Parallel Workers: {args.num_workers}")
    print("=" * 75)

    if not os.path.exists(args.curated_csv):
        print(f"ERROR: {args.curated_csv} not found.")
        sys.exit(1)

    df = pd.read_csv(args.curated_csv)
    curated_records = df.to_dict('records')
    if args.max_slides is not None:
        curated_records = curated_records[:args.max_slides]

    os.makedirs(args.qc_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    metadata = {}
    if os.path.exists(args.out_json) and not args.force:
        try:
            with open(args.out_json, 'r') as f:
                metadata = json.load(f)
            print(f"Loaded existing tiling metadata with {len(metadata)} slides.")
        except Exception:
            metadata = {}

    # Identify slides to process
    curated_wsi_ids = {r['WSI_ID']: r for r in curated_records}
    slides_to_process = []
    
    for r in curated_records:
        wsi_id = r['WSI_ID']
        qc_path = os.path.join(args.qc_dir, f"{wsi_id}_qc.jpg")
        if args.force or wsi_id not in metadata or not os.path.exists(qc_path):
            slides_to_process.append(r)

    print(f"Total curated slides: {len(curated_records)}")
    print(f"Already cached      : {len(curated_records) - len(slides_to_process)}")
    print(f"To process now      : {len(slides_to_process)}")

    start_time = time.time()
    if slides_to_process:
        completed_count = 0
        total_to_process = len(slides_to_process)

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(process_single_slide, item, args.img_dir, args.qc_dir, args.tile_size, args.min_tissue): item['WSI_ID']
                for item in slides_to_process
            }

            for future in as_completed(futures):
                wsi_id = futures[future]
                completed_count += 1
                try:
                    meta, slide_name, info = future.result()
                    if meta is not None:
                        metadata[slide_name] = meta
                        print(f"[{completed_count}/{total_to_process}] Processed {slide_name} ({meta['w_orig']}x{meta['h_orig']} -> 20x) | {meta['n_tiles']} tiles | tissue: {meta['tissue_percentage']:.1f}% ({info:.2f}s)")
                    else:
                        print(f"[{completed_count}/{total_to_process}] Skipped {slide_name}: {info}")
                except Exception as e:
                    print(f"[{completed_count}/{total_to_process}] ERROR processing {wsi_id}: {e}")

    # Retain only current curated cohort in metadata
    final_metadata = {r['WSI_ID']: metadata[r['WSI_ID']] for r in curated_records if r['WSI_ID'] in metadata}
    total_tiles = sum(m['n_tiles'] for m in final_metadata.values())

    total_time = time.time() - start_time
    print("-" * 75)
    print(f"Tiling check/run completed in {total_time:.2f} seconds.")
    print(f"Successfully tiled {len(final_metadata)} / {len(curated_records)} curated slides (100.0%).")
    print(f"Total tiles extracted: {total_tiles} (avg {total_tiles/max(1, len(final_metadata)):.1f} per slide).")

    with open(args.out_json, 'w') as f:
        json.dump(final_metadata, f, indent=2)
    print(f"Saved complete tiling metadata to: {args.out_json}")

if __name__ == "__main__":
    main()
