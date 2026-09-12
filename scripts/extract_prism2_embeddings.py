import os
import sys
import time
import torch
from transformers import AutoModel, AutoConfig

def main():
    print("=" * 80, flush=True)
    print("EXTRACTING PRISM2 EMBEDDINGS (PERCEIVER RESAMPLER & VLM DIAGNOSTIC)", flush=True)
    print("=" * 80, flush=True)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Target device: {device}", flush=True)
    
    tiles_path = "artifacts/features/virchow2_tiles_fp16.pt"
    if not os.path.exists(tiles_path):
        raise FileNotFoundError(f"Virchow2 tile features not found at {tiles_path}")
        
    print(f"Loading cached Virchow2 tile features from {tiles_path}...", flush=True)
    t0 = time.time()
    virchow2_data = torch.load(tiles_path, map_location="cpu")
    print(f"Loaded {len(virchow2_data)} slides in {time.time() - t0:.2f}s.", flush=True)
    
    print("Loading paige-ai/Prism2 with FlashAttention-2 in FP16...", flush=True)
    t0 = time.time()
    cfg = AutoConfig.from_pretrained("paige-ai/Prism2", trust_remote_code=True)
    cfg.text_attn_impl = "flash_attention_2"
    model = AutoModel.from_pretrained(
        "paige-ai/Prism2",
        config=cfg,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device).eval()
    print(f"Prism2 model loaded in {time.time() - t0:.2f}s.", flush=True)
    
    prism2_base_dict = {}         # Perceiver pooled contrastive token (dim=2560)
    prism2_diag_dict = {}         # Phi-3 VLM diagnostic hidden state (dim=3072)
    prism2_latents_mean_dict = {} # Perceiver 256 latents mean-pooled (dim=2560)
    
    wsi_keys = list(virchow2_data.keys())
    n_slides = len(wsi_keys)
    print(f"\nProcessing {n_slides} slides through Prism2 PerceiverResampler and Phi-3 Decoder...", flush=True)
    t_start = time.time()
    
    with torch.no_grad():
        for i, wsi_id in enumerate(wsi_keys):
            slide_item = virchow2_data[wsi_id]
            raw_emb = slide_item["embeddings"] # [N, 2560]
            
            # Virchow2 CLS token is the first 1280 dimensions
            cls_emb = raw_emb[:, :1280].to(device).half().unsqueeze(0) # [1, N, 1280]
            attn_mask = torch.ones((1, cls_emb.shape[1]), dtype=torch.bool, device=device)
            
            # 1. Perceiver pooled contrastive token (dim=2560)
            resampler_out = model._encode_images(cls_emb, attn_mask)
            token_name = model.config.pooler_tokens[0]
            base_emb = resampler_out[token_name][:, 0].squeeze(0).cpu().half()
            prism2_base_dict[wsi_id] = base_emb
            
            # 2. Perceiver latents mean (dim=2560)
            latents = resampler_out["latents"] # [1, 256, 2560]
            lat_mean = latents.mean(dim=1).squeeze(0).cpu().half()
            prism2_latents_mean_dict[wsi_id] = lat_mean
            
            # 3. Phi-3 VLM diagnostic embedding (dim=3072)
            diag_emb = model.get_diagnostic_embedding(cls_emb, attn_mask).squeeze(0).cpu().half()
            prism2_diag_dict[wsi_id] = diag_emb
            
            if (i + 1) % 20 == 0 or (i + 1) == n_slides:
                elapsed = time.time() - t_start
                rate = elapsed / (i + 1)
                remaining = rate * (n_slides - (i + 1))
                print(f"[{i + 1}/{n_slides}] {wsi_id} processed | Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s", flush=True)
            
    total_time = time.time() - t_start
    print(f"\nExtraction complete in {total_time:.2f}s ({total_time / n_slides:.3f}s per slide).", flush=True)
    
    os.makedirs("artifacts/features", exist_ok=True)
    out_base = "artifacts/features/prism2_base_slide.pt"
    out_diag = "artifacts/features/prism2_diag_slide.pt"
    out_lat_mean = "artifacts/features/prism2_latents_mean_slide.pt"
    
    print(f"Saving {out_base} (dim={prism2_base_dict[wsi_keys[0]].shape[-1]})...", flush=True)
    torch.save(prism2_base_dict, out_base)
    
    print(f"Saving {out_diag} (dim={prism2_diag_dict[wsi_keys[0]].shape[-1]})...", flush=True)
    torch.save(prism2_diag_dict, out_diag)
    
    print(f"Saving {out_lat_mean} (dim={prism2_latents_mean_dict[wsi_keys[0]].shape[-1]})...", flush=True)
    torch.save(prism2_latents_mean_dict, out_lat_mean)
    
    print("\nSUCCESS: All Prism2 representations saved to disk!", flush=True)

if __name__ == "__main__":
    main()
