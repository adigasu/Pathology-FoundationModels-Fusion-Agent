import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# -----------------------------------------------------------------------------
# Early Fusion Pipeline
# -----------------------------------------------------------------------------
class EarlyFusionClassifier:
    def __init__(
        self,
        c: float = 1.0,
        l2_norm_per_stream: bool = False,
        pca_dim: Optional[int] = None,
        use_metadata: bool = True,
        class_weight: Optional[str] = None,
        random_state: int = 42
    ):
        self.c = c
        self.l2_norm_per_stream = l2_norm_per_stream
        self.pca_dim = pca_dim
        self.use_metadata = use_metadata
        self.class_weight = class_weight
        self.random_state = random_state
        self.pca = None
        self.clf = None

    def _prepare_features(
        self,
        features: Dict[str, np.ndarray],
        metadata: np.ndarray,
        is_train: bool = True
    ) -> np.ndarray:
        streams = []
        for name in ["uni2", "virchow2", "gigapath"]:
            x = features[name].copy()
            if self.l2_norm_per_stream:
                norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
                x = x / norm
            streams.append(x)
        concat_vis = np.concatenate(streams, axis=1)

        if self.pca_dim is not None:
            if is_train:
                # Fit PCA on training split only (Zero Leakage)
                max_components = min(self.pca_dim, concat_vis.shape[0] - 1, concat_vis.shape[1])
                self.pca = PCA(n_components=max_components, whiten=True, random_state=self.random_state)
                vis_proj = self.pca.fit_transform(concat_vis)
            else:
                vis_proj = self.pca.transform(concat_vis)
        else:
            vis_proj = concat_vis

        if self.use_metadata:
            return np.concatenate([vis_proj, metadata], axis=1)
        return vis_proj

    def fit(self, features: Dict[str, np.ndarray], metadata: np.ndarray, y: np.ndarray):
        x_train = self._prepare_features(features, metadata, is_train=True)
        self.clf = LogisticRegression(
            C=self.c,
            penalty="l2",
            solver="lbfgs",
            max_iter=150, tol=1e-3,
            class_weight=self.class_weight,
            random_state=self.random_state
        )
        self.clf.fit(x_train, y)
        return self

    def predict_proba(self, features: Dict[str, np.ndarray], metadata: np.ndarray) -> np.ndarray:
        x_val = self._prepare_features(features, metadata, is_train=False)
        return self.clf.predict_proba(x_val)


# -----------------------------------------------------------------------------
# Late Fusion Pipeline
# -----------------------------------------------------------------------------
class LateFusionClassifier:
    def __init__(
        self,
        strategy: str = "uniform",  # 'uniform', 'temperature', 'simplex', 'stacking'
        c: float = 1.0,
        tau: float = 1.5,
        use_metadata: bool = True,
        class_weight: Optional[str] = None,
        random_state: int = 42
    ):
        self.strategy = strategy
        self.c = c
        self.tau = tau
        self.use_metadata = use_metadata
        self.class_weight = class_weight
        self.random_state = random_state
        self.stream_clfs: Dict[str, LogisticRegression] = {}
        self.meta_clf: Optional[LogisticRegression] = None
        self.simplex_weights: Optional[np.ndarray] = None
        self.stacking_meta_learner: Optional[LogisticRegression] = None

    def _prepare_stream_feature(self, x: np.ndarray, metadata: np.ndarray) -> np.ndarray:
        if self.use_metadata:
            return np.concatenate([x, metadata], axis=1)
        return x

    def fit(self, features: Dict[str, np.ndarray], metadata: np.ndarray, y: np.ndarray):
        model_names = ["uni2", "virchow2", "gigapath"]
        
        # Train individual stream base classifiers
        for name in model_names:
            x_m = self._prepare_stream_feature(features[name], metadata)
            clf = LogisticRegression(
                C=self.c,
                penalty="l2",
                solver="lbfgs",
                max_iter=150, tol=1e-3,
                class_weight=self.class_weight,
                random_state=self.random_state
            )
            clf.fit(x_m, y)
            self.stream_clfs[name] = clf

        if self.strategy == "simplex":
            # Optimize simplex weights on training set probabilities using constrained optimization
            probas = []
            for name in model_names:
                x_m = self._prepare_stream_feature(features[name], metadata)
                probas.append(self.stream_clfs[name].predict_proba(x_m))
            
            # One-hot encoded y
            n_samples = len(y)
            n_classes = len(np.unique(y))
            y_onehot = np.zeros((n_samples, 7))
            y_onehot[np.arange(n_samples), y] = 1.0

            def loss_fn(weights):
                w = np.array(weights).reshape(3, 1, 1)
                p_comb = (w[0] * probas[0] + w[1] * probas[1] + w[2] * probas[2])
                p_comb = np.clip(p_comb, 1e-12, 1.0 - 1e-12)
                return -np.mean(np.sum(y_onehot * np.log(p_comb), axis=1))

            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
            bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
            init_w = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
            res = minimize(loss_fn, init_w, method='SLSQP', bounds=bounds, constraints=cons)
            self.simplex_weights = res.x if res.success else np.array(init_w)

        elif self.strategy == "stacking":
            # Out-of-fold stacking meta-learner
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            oof_probas = [np.zeros((len(y), 7)) for _ in model_names]

            for tr_idx, val_idx in skf.split(features["uni2"], y):
                for m_idx, name in enumerate(model_names):
                    x_m = self._prepare_stream_feature(features[name], metadata)
                    fold_clf = LogisticRegression(
                        C=self.c,
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=150, tol=1e-3,
                        class_weight=self.class_weight,
                        random_state=self.random_state
                    )
                    fold_clf.fit(x_m[tr_idx], y[tr_idx])
                    oof_probas[m_idx][val_idx] = fold_clf.predict_proba(x_m[val_idx])

            # Stacking features: concat OOF probabilities (+ metadata if enabled)
            stack_train = np.concatenate(oof_probas, axis=1)
            if self.use_metadata:
                stack_train = np.concatenate([stack_train, metadata], axis=1)

            self.stacking_meta_learner = LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="lbfgs",
                max_iter=150, tol=1e-3,
                random_state=self.random_state
            )
            self.stacking_meta_learner.fit(stack_train, y)

        return self

    def predict_proba(self, features: Dict[str, np.ndarray], metadata: np.ndarray) -> np.ndarray:
        model_names = ["uni2", "virchow2", "gigapath"]
        probas = []
        for name in model_names:
            x_m = self._prepare_stream_feature(features[name], metadata)
            probas.append(self.stream_clfs[name].predict_proba(x_m))

        if self.strategy == "uniform":
            return (probas[0] + probas[1] + probas[2]) / 3.0

        elif self.strategy == "temperature":
            # Temperature-scaled logit voting
            scaled_logits = []
            for p in probas:
                p_safe = np.clip(p, 1e-12, 1.0)
                logits = np.log(p_safe) / self.tau
                scaled_logits.append(logits)
            mean_logits = np.mean(scaled_logits, axis=0)
            # Softmax
            exp_logits = np.exp(mean_logits - np.max(mean_logits, axis=1, keepdims=True))
            return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        elif self.strategy == "simplex":
            w = self.simplex_weights
            return w[0] * probas[0] + w[1] * probas[1] + w[2] * probas[2]

        elif self.strategy == "stacking":
            stack_test = np.concatenate(probas, axis=1)
            if self.use_metadata:
                stack_test = np.concatenate([stack_test, metadata], axis=1)
            return self.stacking_meta_learner.predict_proba(stack_test)

        raise ValueError(f"Unknown late fusion strategy: {self.strategy}")


# -----------------------------------------------------------------------------
# Intermediate Fusion Modules (Fast PyTorch, <1500 params)
# -----------------------------------------------------------------------------
class GatedFusionModule(nn.Module):
    def __init__(self, in_dims: List[int], proj_dim: int = 64, num_classes: int = 7, use_meta: bool = True, dropout: float = 0.2):
        super().__init__()
        self.use_meta = use_meta
        self.projs = nn.ModuleList([nn.Linear(dim, proj_dim, bias=False) for dim in in_dims])
        self.gate = nn.Linear(proj_dim * 3, 3)
        self.dropout = nn.Dropout(dropout)
        meta_dim = 2 if use_meta else 0
        self.head = nn.Linear(proj_dim + meta_dim, num_classes)

    def forward(self, xs: List[torch.Tensor], meta: Optional[torch.Tensor] = None) -> torch.Tensor:
        # xs: list of 3 tensors [B, D_m]
        projected = [proj(x) for proj, x in zip(self.projs, xs)] # 3 x [B, proj_dim]
        proj_cat = torch.cat(projected, dim=-1)                   # [B, proj_dim * 3]
        gates = torch.softmax(self.gate(proj_cat), dim=-1)        # [B, 3]

        # Weighted combination: \sum_m alpha_m * h_m
        fused = (
            gates[:, 0:1] * projected[0] +
            gates[:, 1:2] * projected[1] +
            gates[:, 2:3] * projected[2]
        )
        fused = self.dropout(fused)
        if self.use_meta and meta is not None:
            fused = torch.cat([fused, meta], dim=-1)
        return self.head(fused)


class StreamABMILModule(nn.Module):
    def __init__(self, in_dims: List[int], proj_dim: int = 64, num_classes: int = 7, use_meta: bool = True):
        super().__init__()
        self.use_meta = use_meta
        self.projs = nn.ModuleList([nn.Linear(dim, proj_dim) for dim in in_dims])
        self.ln = nn.LayerNorm(proj_dim)
        self.attn_v = nn.Linear(proj_dim, 32)
        self.attn_w = nn.Linear(32, 1, bias=False)
        meta_dim = 2 if use_meta else 0
        self.head = nn.Linear(proj_dim + meta_dim, num_classes)

    def forward(self, xs: List[torch.Tensor], meta: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Project each stream into common space: [B, 3, proj_dim]
        projected = torch.stack([self.ln(proj(x)) for proj, x in zip(self.projs, xs)], dim=1)
        # Attention across 3 streams
        a = torch.tanh(self.attn_v(projected)) # [B, 3, 32]
        a = self.attn_w(a).squeeze(-1)         # [B, 3]
        attn_weights = torch.softmax(a, dim=-1).unsqueeze(-1) # [B, 3, 1]

        pooled = torch.sum(attn_weights * projected, dim=1)   # [B, proj_dim]
        if self.use_meta and meta is not None:
            pooled = torch.cat([pooled, meta], dim=-1)
        return self.head(pooled)


class IntermediateFusionClassifier:
    def __init__(
        self,
        arch: str = "gated",  # 'gated', 'abmil', 'pca_gated'
        pca_per_stream: Optional[int] = None,
        use_metadata: bool = True,
        dropout: float = 0.2,
        weight_decay: float = 1e-3,
        lr: float = 1e-2,
        epochs: int = 60,
        class_weight: Optional[str] = None,
        random_state: int = 42
    ):
        self.arch = arch
        self.pca_per_stream = pca_per_stream
        self.use_metadata = use_metadata
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.epochs = epochs
        self.class_weight = class_weight
        self.random_state = random_state
        self.pcas: Dict[str, PCA] = {}
        self.module: Optional[nn.Module] = None

    def _prepare_stream_inputs(
        self,
        features: Dict[str, np.ndarray],
        metadata: np.ndarray,
        is_train: bool = True
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        xs = []
        for name in ["uni2", "virchow2", "gigapath"]:
            x = features[name].copy()
            if self.pca_per_stream is not None:
                if is_train:
                    pca = PCA(n_components=min(self.pca_per_stream, x.shape[0]-1, x.shape[1]), whiten=True, random_state=self.random_state)
                    x = pca.fit_transform(x)
                    self.pcas[name] = pca
                else:
                    x = self.pcas[name].transform(x)
            xs.append(torch.from_numpy(x).float())
        meta_t = torch.from_numpy(metadata).float()
        return xs, meta_t

    def fit(self, features: Dict[str, np.ndarray], metadata: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.random_state)
        xs, meta_t = self._prepare_stream_inputs(features, metadata, is_train=True)
        y_t = torch.from_numpy(y).long()

        in_dims = [x.shape[1] for x in xs]
        if self.arch in ("gated", "pca_gated"):
            self.module = GatedFusionModule(in_dims=in_dims, proj_dim=64, num_classes=7, use_meta=self.use_metadata, dropout=self.dropout)
        elif self.arch == "abmil":
            self.module = StreamABMILModule(in_dims=in_dims, proj_dim=64, num_classes=7, use_meta=self.use_metadata)
        else:
            raise ValueError(f"Unknown intermediate architecture: {self.arch}")

        # Class weights if requested
        if self.class_weight == "balanced":
            counts = np.bincount(y, minlength=7)
            weights = len(y) / (7.0 * np.maximum(counts, 1))
            criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).float())
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(self.module.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.module.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            logits = self.module(xs, meta_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

        return self

    def predict_proba(self, features: Dict[str, np.ndarray], metadata: np.ndarray) -> np.ndarray:
        self.module.eval()
        with torch.no_grad():
            xs, meta_t = self._prepare_stream_inputs(features, metadata, is_train=False)
            logits = self.module(xs, meta_t)
            probas = torch.softmax(logits, dim=-1).cpu().numpy()
        return probas
