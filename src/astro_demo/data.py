"""Synthetic star field generation for the astrometry demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


@dataclass
class StarField:
    """Simple container for synthetic star field metadata."""

    tile_id: int
    ra_deg: float
    dec_deg: float
    pixel_coords: np.ndarray  # shape (n_stars, 2), values in [0, 1]
    magnitudes: np.ndarray  # shape (n_stars,)


def _sample_unit_disk(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample n (x, y) positions from a mildly clustered distribution."""
    radii = rng.power(4, size=n)  # bias towards center
    angles = rng.uniform(0.0, 2 * np.pi, size=n)
    x = 0.5 + radii * np.cos(angles) * 0.45
    y = 0.5 + radii * np.sin(angles) * 0.45
    return np.stack([x.clip(0, 1), y.clip(0, 1)], axis=1)


def _sample_magnitudes(rng: np.random.Generator, n: int) -> np.ndarray:
    """Create log-uniform magnitudes to mimic bright vs faint stars."""
    mags = rng.uniform(0.0, 2.0, size=n)
    return np.sort(mags)  # brighter stars first


def generate_catalog(
    num_tiles: int,
    stars_per_tile: Tuple[int, int] = (12, 24),
    seed: int | None = 17,
) -> List[StarField]:
    """Generate a toy sky catalog with random RA/Dec and pixel star positions."""

    rng = np.random.default_rng(seed)
    catalog: List[StarField] = []

    for tile_id in range(num_tiles):
        ra_deg = rng.uniform(0.0, 360.0)
        dec_deg = rng.uniform(-90.0, 90.0)
        n_stars = rng.integers(stars_per_tile[0], stars_per_tile[1] + 1)
        coords = _sample_unit_disk(rng, n_stars)
        magnitudes = _sample_magnitudes(rng, n_stars)
        catalog.append(
            StarField(
                tile_id=tile_id,
                ra_deg=float(ra_deg),
                dec_deg=float(dec_deg),
                pixel_coords=coords,
                magnitudes=magnitudes,
            )
        )

    return catalog


def render_star_field(
    field: StarField,
    image_size: int = 128,
    psf_sigma: float = 1.5,
    noise_level: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Render a PSF-blurred image of the star field."""

    rng = rng or np.random.default_rng()
    xv, yv = np.meshgrid(
        np.linspace(0.0, 1.0, image_size, endpoint=False),
        np.linspace(0.0, 1.0, image_size, endpoint=False),
    )
    image = np.zeros((image_size, image_size), dtype=np.float32)

    for (x, y), mag in zip(field.pixel_coords, field.magnitudes):
        flux = np.exp(-mag)  # brighter stars have lower magnitude
        dx = xv - x
        dy = yv - y
        exponent = -(dx**2 + dy**2) / (2 * (psf_sigma / image_size) ** 2)
        image += flux * np.exp(exponent)

    noise = rng.normal(scale=noise_level, size=image.shape).astype(np.float32)
    image = image + noise
    image -= image.min()
    image /= image.max() + 1e-8
    return image


def mask_star_field(
    field: StarField,
    min_visible: int = 3,
    max_visible: int = 7,
    seed: int | None = None,
) -> StarField:
    """Return a copy of the field with most stars masked out."""

    rng = np.random.default_rng(seed)
    n_visible = min(len(field.pixel_coords), rng.integers(min_visible, max_visible + 1))
    idx = np.arange(len(field.pixel_coords))
    rng.shuffle(idx)
    keep = np.sort(idx[:n_visible])

    return StarField(
        tile_id=field.tile_id,
        ra_deg=field.ra_deg,
        dec_deg=field.dec_deg,
        pixel_coords=field.pixel_coords[keep],
        magnitudes=field.magnitudes[keep],
    )


def iter_dataset(
    catalog: Iterable[StarField],
    image_size: int = 128,
    min_visible: int = 3,
    max_visible: int = 7,
    seed: int | None = None,
) -> Iterable[Tuple[np.ndarray, StarField]]:
    """Yield (image, metadata) pairs with random masking per tile."""

    rng = np.random.default_rng(seed)
    for field in catalog:
        masked = mask_star_field(
            field,
            min_visible=min_visible,
            max_visible=max_visible,
            seed=rng.integers(0, 2**32 - 1),
        )
        image = render_star_field(
            masked,
            image_size=image_size,
            rng=np.random.default_rng(rng.integers(0, 2**32 - 1)),
        )
        yield image, masked
