import os
import time
import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

class WSITiler:
    def __init__(self, tile_size: int = 224, min_tissue_frac: float = 0.50, thumb_factor: int = 32):
        self.tile_size = tile_size
        self.min_tissue_frac = min_tissue_frac
        self.thumb_factor = thumb_factor

    def process_slide(self, wsi_path: str):
        """
        Processes an 80x WSI:
        1. Reads thumbnail at 1/thumb_factor scale for fast tissue segmentation.
        2. Detects tissue regions rejecting glass background.
        3. Generates 20x non-overlapping tile coordinates.
        """
        with Image.open(wsi_path) as img:
            w_orig, h_orig = img.size
            thumb_w = max(1, w_orig // self.thumb_factor)
            thumb_h = max(1, h_orig // self.thumb_factor)
            
            # Fast JPEG draft decoding
            img.draft('RGB', (thumb_w, thumb_h))
            thumb_img = img.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)

        # Convert thumbnail to numpy for segmentation
        thumb_arr = np.array(thumb_img)
        gray = np.mean(thumb_arr, axis=2)
        
        # Robust tissue detection:
        # Glass background is bright (gray > 220) and low-variance across channels
        std_dev = np.std(thumb_arr, axis=2)
        tissue_mask = (gray < 220) & (std_dev > 5)
        tissue_frac = float(np.mean(tissue_mask))

        # 20x dimensions (4x linear downsample from 80x)
        w_20x = w_orig // 4
        h_20x = h_orig // 4

        scale_x = thumb_w / w_20x
        scale_y = thumb_h / h_20x

        valid_tiles_20x = []
        for y in range(0, h_20x - self.tile_size + 1, self.tile_size):
            for x in range(0, w_20x - self.tile_size + 1, self.tile_size):
                # Map to thumbnail coordinates
                tx1 = int(x * scale_x)
                ty1 = int(y * scale_y)
                tx2 = max(tx1 + 1, int((x + self.tile_size) * scale_x))
                ty2 = max(ty1 + 1, int((y + self.tile_size) * scale_y))

                tile_tissue_cov = np.mean(tissue_mask[ty1:ty2, tx1:tx2])
                if tile_tissue_cov >= self.min_tissue_frac:
                    valid_tiles_20x.append({
                        'x_20x': int(x),
                        'y_20x': int(y),
                        'w_20x': int(self.tile_size),
                        'h_20x': int(self.tile_size),
                        'tissue_coverage': float(tile_tissue_cov)
                    })

        return {
            'w_orig': w_orig,
            'h_orig': h_orig,
            'w_20x': w_20x,
            'h_20x': h_20x,
            'tissue_percentage': round(tissue_frac * 100, 2),
            'n_tiles': len(valid_tiles_20x),
            'tiles': valid_tiles_20x,
            'thumb_img': thumb_img,
            'thumb_mask': tissue_mask,
            'scale_x': scale_x,
            'scale_y': scale_y
        }

    def save_qc_overlay(self, result: dict, out_path: str):
        """
        Renders and saves a diagnostic QC overlay thumbnail showing selected tiles.
        """
        thumb = result['thumb_img'].copy()
        draw = ImageDraw.Draw(thumb, 'RGBA')
        
        scale_x = result['scale_x']
        scale_y = result['scale_y']

        for t in result['tiles']:
            tx1 = int(t['x_20x'] * scale_x)
            ty1 = int(t['y_20x'] * scale_y)
            tx2 = int((t['x_20x'] + t['w_20x']) * scale_x)
            ty2 = int((t['y_20x'] + t['h_20x']) * scale_y)
            draw.rectangle([tx1, ty1, tx2, ty2], outline=(0, 230, 115, 200), width=1)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        thumb.save(out_path, quality=85)
