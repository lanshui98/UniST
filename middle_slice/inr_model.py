"""Lightweight Fourier-feature INR for middle-slice gene expression."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA


def _build_ffn(dim_in: int, dim_out: int, dim_hidden: int = 512, n_layers: int = 3):
    import torch
    import torch.nn as nn

    class GaussianEncoding(nn.Module):
        def __init__(self, num_input_channels, mapping_size=128, scale=10.0):
            super().__init__()
            B = torch.randn((num_input_channels, mapping_size)) * scale
            self.register_buffer("B", B)

        def forward(self, x):
            x = x @ self.B
            x = 2 * np.pi * x
            return torch.cat([torch.sin(x), torch.cos(x)], dim=1)

    class EnhancedEncoding(nn.Module):
        def __init__(self, mapping_size=64, scales=(1.0, 10.0, 100.0), z_scales=(0.1, 1.0, 10.0)):
            super().__init__()
            self.xy = nn.ModuleList([GaussianEncoding(2, mapping_size, s) for s in scales])
            self.z = nn.ModuleList([GaussianEncoding(1, mapping_size, s) for s in z_scales])

        def forward(self, x):
            xy = x[:, :2]
            z = x[:, 2:3]
            parts = [enc(xy) for enc in self.xy] + [enc(z) for enc in self.z]
            return torch.cat(parts, dim=1)

    class FourierFeatureNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoding = EnhancedEncoding()
            # 3 scales * 2 (sin/cos) * mapping_size for xy + same for z
            enc_dim = 3 * 2 * 64 + 3 * 2 * 64  # 768
            layers = [nn.Linear(enc_dim, dim_hidden), nn.ReLU()]
            for _ in range(n_layers - 1):
                layers += [nn.Linear(dim_hidden, dim_hidden), nn.ReLU()]
            layers.append(nn.Linear(dim_hidden, dim_out))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(self.encoding(x))

    return FourierFeatureNet()


def fit_predict_inr(
    train_coords: np.ndarray,
    train_X: np.ndarray,
    query_coords: np.ndarray,
    *,
    n_pcs: Optional[int] = 50,
    epochs: int = 300,
    batch_size: int = 1024,
    lr: float = 1e-3,
    device: Optional[str] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, dict]:
    """
    Train a lightweight FFN-INR: normalized coords → expression (or PCA space).

    Returns predicted expression for query_coords and a small info dict.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as e:
        raise ImportError(
            "INR mode requires PyTorch. Install with: pip install torch"
        ) from e

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pca = None
    y = train_X.astype(np.float32)
    if n_pcs is not None and y.shape[1] > n_pcs:
        n_comp = min(n_pcs, y.shape[0] - 1, y.shape[1])
        pca = PCA(n_components=n_comp, random_state=seed)
        y = pca.fit_transform(y).astype(np.float32)

    model = _build_ffn(dim_in=3, dim_out=y.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    coords_t = torch.from_numpy(train_coords.astype(np.float32))
    y_t = torch.from_numpy(y)

    n = coords_t.shape[0]
    model.train()
    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = coords_t[idx].to(device)
            yb = y_t[idx].to(device)
            pred = model(xb)
            loss = F.mse_loss(pred, yb) + F.l1_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        q = torch.from_numpy(query_coords.astype(np.float32)).to(device)
        preds = []
        for start in range(0, q.shape[0], batch_size):
            preds.append(model(q[start : start + batch_size]).cpu().numpy())
        pred_y = np.vstack(preds)

    if pca is not None:
        pred_X = pca.inverse_transform(pred_y).astype(np.float32)
    else:
        pred_X = pred_y.astype(np.float32)

    info = {
        "device": device,
        "epochs": epochs,
        "n_pcs": None if pca is None else int(pca.n_components_),
        "n_genes": int(train_X.shape[1]),
    }
    return pred_X, info
