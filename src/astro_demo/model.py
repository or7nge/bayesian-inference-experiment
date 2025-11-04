"""Lightweight transformer baseline for sparse asterisms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .data import StarField


def _pad_stars(
    field: StarField,
    max_stars: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad star positions/magnitudes to a fixed length tensor."""

    coords = torch.tensor(field.pixel_coords, dtype=torch.float32)
    mags = torch.tensor(field.magnitudes[:, None], dtype=torch.float32)
    features = torch.cat([coords, mags], dim=1)

    n = features.shape[0]
    if n >= max_stars:
        return features[:max_stars], torch.zeros(max_stars, dtype=torch.bool)

    pad = torch.zeros((max_stars - n, 3), dtype=torch.float32)
    mask = torch.zeros(max_stars, dtype=torch.bool)
    mask[n:] = True
    features = torch.cat([features, pad], dim=0)
    return features, mask


class SparseAsterismDataset(Dataset):
    """Simple in-memory dataset of sparse star fields."""

    def __init__(
        self,
        samples: Iterable[Tuple[StarField, int]],
        *,
        max_stars: int,
    ) -> None:
        self._items: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
        for field, label in samples:
            feats, mask = _pad_stars(field, max_stars=max_stars)
            self._items.append((feats, mask.bool(), label))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        return self._items[idx]


class StarPositionalEncoding(nn.Module):
    """Sinusoidal encoding for normalized sky positions."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (batch, n_stars, 2)
        half = self.dim // 2
        freqs = torch.arange(half, device=coords.device, dtype=coords.dtype)
        freqs = 1.0 / (1000 ** (freqs / half))
        angle = coords.unsqueeze(-1) * freqs
        sin = torch.sin(angle)
        cos = torch.cos(angle)
        enc = torch.cat([sin, cos], dim=-1)
        return enc.flatten(-2)


class ToyAsterismTransformer(nn.Module):
    """Minimal transformer encoder for classifying sparse asterisms."""

    def __init__(
        self,
        *,
        max_stars: int,
        feature_dim: int = 3,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_stars = max_stars
        self.num_classes = num_classes

        self.pos_encoder = StarPositionalEncoding(dim=d_model // 2)
        self.feature_proj = nn.Linear(feature_dim + d_model, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                batch_first=True,
            ), 
            num_layers=num_layers
        )
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute class logits for each batch example."""

        # features: (batch, max_stars, 3), padding_mask: (batch, max_stars)
        coords = features[..., :2]
        mag = features[..., 2:]
        pos = self.pos_encoder(coords)
        tokens = torch.cat([coords, mag, pos], dim=-1)
        tokens = self.feature_proj(tokens)
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        pooled = encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0).sum(dim=1)
        denom = (~padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
        pooled = pooled / denom
        return self.cls_head(pooled)


@dataclass
class TrainingConfig:
    """Container for hyperparameters."""

    max_stars: int = 12
    num_classes: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    epochs: int = 30
    batch_size: int = 64


def make_model(config: TrainingConfig) -> ToyAsterismTransformer:
    """Instantiate the transformer with the given hyperparameters."""

    return ToyAsterismTransformer(
        max_stars=config.max_stars,
        num_classes=config.num_classes,
        dropout=config.dropout,
    )


def classification_step(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute loss and predictions for a batch."""

    features, padding_mask, labels = batch
    logits = model(features, padding_mask)
    loss = F.cross_entropy(logits, labels)
    return loss, logits


def mc_dropout_predictions(
    model: nn.Module,
    features: torch.Tensor,
    padding_mask: torch.Tensor,
    *,
    passes: int = 20,
) -> torch.Tensor:
    """Collect Monte-Carlo dropout predictions for uncertainty estimates."""

    model.train()  # enable dropout
    preds = []
    for _ in range(passes):
        with torch.no_grad():
            logits = model(features, padding_mask)
            preds.append(F.softmax(logits, dim=-1))
    stacked = torch.stack(preds, dim=0)
    model.eval()
    return stacked
