import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from sklearn.model_selection import StratifiedKFold

CLASS_NAMES = ['Acinar', 'Cribriform', 'In situ', 'Lepidic', 'Micropapillary', 'Papillary', 'Solid']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASS_NAMES)}

def load_cached_embeddings(
    feature_dir: str = "artifacts/features",
    models: List[str] = ("uni2", "virchow2", "gigapath"),
    pooling: str = "mean"
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Loads cached slide-level embeddings from disk.
    Supports pooling = 'mean', 'max', or 'concat' ([mean; max]).
    Also supports Prism2 embeddings (prism2_diag, prism2_base, prism2_latents_mean).
    """
    embeddings = {}
    for model_name in models:
        # Check for direct slide-level representations (e.g. Prism2 VLM / Perceiver)
        direct_path = os.path.join(feature_dir, f"{model_name}_slide.pt")
        if model_name == "prism2" and not os.path.exists(direct_path):
            direct_path = os.path.join(feature_dir, "prism2_diag_slide.pt")

        if os.path.exists(direct_path):
            feat_dict = torch.load(direct_path, map_location="cpu", weights_only=False)
            embeddings[model_name] = feat_dict
            continue

        mean_path = os.path.join(feature_dir, f"{model_name}_slide_mean.pt")
        max_path = os.path.join(feature_dir, f"{model_name}_slide_max.pt")
        
        if pooling == "mean":
            feat_dict = torch.load(mean_path, map_location="cpu", weights_only=False)
        elif pooling == "max":
            feat_dict = torch.load(max_path, map_location="cpu", weights_only=False)
        elif pooling in ("concat", "mean_max"):
            mean_dict = torch.load(mean_path, map_location="cpu", weights_only=False)
            max_dict = torch.load(max_path, map_location="cpu", weights_only=False)
            feat_dict = {
                wsi_id: torch.cat([mean_dict[wsi_id].float(), max_dict[wsi_id].float()], dim=-1)
                for wsi_id in mean_dict
            }
        elif pooling in ("top5", "top_5"):
            p_path = os.path.join(feature_dir, f"{model_name}_slide_top5.pt")
            feat_dict = torch.load(p_path, map_location="cpu", weights_only=False)
        elif pooling in ("top10", "top_10"):
            p_path = os.path.join(feature_dir, f"{model_name}_slide_top10.pt")
            feat_dict = torch.load(p_path, map_location="cpu", weights_only=False)
        elif pooling in ("top20", "top_20"):
            p_path = os.path.join(feature_dir, f"{model_name}_slide_top20.pt")
            feat_dict = torch.load(p_path, map_location="cpu", weights_only=False)
        else:
            raise ValueError(f"Unknown pooling mode: {pooling}")
        embeddings[model_name] = feat_dict
    return embeddings

def load_multimodal_dataset(
    curated_csv: str = "artifacts/curated_patients.csv",
    splits_json: Optional[str] = None,
    seed: int = 42,
    gamma_meta: float = 1.0,
    pooling: str = "mean",
    feature_dir: str = "artifacts/features",
    models: List[str] = ("uni2", "virchow2", "gigapath")
) -> Dict[str, Any]:
    """
    Loads patient cohort, metadata, and slide embeddings with strict zero-leakage guarantee.
    Age standardization statistics (mu, sigma) are computed strictly on the training set.
    Supports selecting arbitrary subsets of foundation models, including Prism2.
    """
    if splits_json is None:
        splits_json = f"artifacts/splits/splits_seed_{seed}.json"
    
    if not os.path.exists(curated_csv):
        raise FileNotFoundError(f"Curated patients CSV not found at {curated_csv}")
    if not os.path.exists(splits_json):
        raise FileNotFoundError(f"Splits JSON not found at {splits_json}")
        
    df = pd.read_csv(curated_csv)
    with open(splits_json, 'r') as f:
        split_data = json.load(f)
    patient_splits = split_data["patient_splits"]
    
    # Load embeddings
    raw_feats = load_cached_embeddings(feature_dir=feature_dir, models=models, pooling=pooling)
    
    # Order patients consistently
    df['split'] = df['patient_id'].map(lambda p: patient_splits[p]['split'])
    df['y'] = df['Tumor Subtype'].map(CLASS_TO_IDX)
    
    # Compute Train-only Age statistics (Strict Zero-Leakage)
    train_mask = (df['split'] == 'train').values
    train_ages = df.loc[train_mask, 'Age'].values.astype(np.float32)
    age_mu = float(np.mean(train_ages))
    age_sigma = float(np.std(train_ages)) if float(np.std(train_ages)) > 1e-6 else 1.0
    
    # Standardize Age and encode Sex (Female=1.0, Male=0.0)
    norm_age = (df['Age'].values.astype(np.float32) - age_mu) / age_sigma
    sex_binary = np.where(df['Sex'].values == 'Female', 1.0, 0.0).astype(np.float32)
    
    # Combined metadata matrix scaled by gamma_meta
    meta_array = np.column_stack([norm_age, sex_binary]) * float(gamma_meta)
    
    # Extract features per model
    n_patients = len(df)
    features = {}
    for model_name, feat_dict in raw_feats.items():
        dim = feat_dict[df.iloc[0]['WSI_ID']].shape[-1]
        feat_matrix = np.zeros((n_patients, dim), dtype=np.float32)
        for i, row in df.iterrows():
            wsi_id = row['WSI_ID']
            feat_matrix[i] = feat_dict[wsi_id].cpu().numpy().astype(np.float32)
        features[model_name] = feat_matrix
        
    splits_mask = {
        'train': (df['split'] == 'train').values,
        'val': (df['split'] == 'val').values,
        'test': (df['split'] == 'test').values,
        'train_val': (df['split'].isin(['train', 'val'])).values
    }
    
    return {
        'df': df,
        'patient_ids': df['patient_id'].tolist(),
        'wsi_ids': df['WSI_ID'].tolist(),
        'subtypes': df['Tumor Subtype'].tolist(),
        'y': df['y'].values.astype(np.int64),
        'metadata': meta_array,
        'features': features,
        'splits_mask': splits_mask,
        'meta_stats': {'age_mu': age_mu, 'age_sigma': age_sigma},
        'seed': seed,
        'gamma_meta': gamma_meta,
        'pooling': pooling,
        'models': list(models)
    }

def get_train_val_cv_folds(
    dataset: Dict[str, Any],
    n_splits: int = 5,
    cv_seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Constructs 5-fold Stratified CV splits on the Train+Val cohort (n=163).
    Returns list of (train_idx, val_idx) as integer indices indexing dataset['splits_mask']['train_val'].
    """
    tv_indices = np.where(dataset['splits_mask']['train_val'])[0]
    tv_y = dataset['y'][tv_indices]
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)
    folds = []
    for tr_sub_idx, va_sub_idx in skf.split(tv_indices, tv_y):
        tr_idx = tv_indices[tr_sub_idx]
        va_idx = tv_indices[va_sub_idx]
        folds.append((tr_idx, va_idx))
    return folds
