"""Interop with the Astrometry.net CLI for the demo."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np


def _default_args() -> list[str]:
    """Default arguments that keep runtime reasonable on tiny images."""

    return [
        "--downsample",
        "2",
        "--tweak-order",
        "2",
        # "--use-sextractor",
        "--scale-units",
        "arcsecperpix",
        "--scale-low",
        "20",
        "--scale-high",
        "120",
    ]


def call_solve_field(
    image: np.ndarray,
    *,
    extra_args: Optional[Iterable[str]] = None,
    keep_intermediate: bool = False,
) -> Dict[str, float]:
    """Run astrometry.net's solve-field on a numpy image.

    Parameters
    ----------
    image:
        Normalized image array (values in [0, 1]).
    extra_args:
        Additional CLI flags for solve-field.
    keep_intermediate:
        When True, leave the temporary FITS and solution files on disk for inspection.
    """

    image = (image * 2**16).astype(np.uint16)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        fits_path = tmp_path / "input.fits"
        _write_fits(image, fits_path)

        args = ["solve-field", str(fits_path), "--no-plots", "--overwrite"]
        args.extend(_default_args())
        if extra_args:
            args.extend(extra_args)

        result = subprocess.run(
            args,
            cwd=tmp_path,
            check=False,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"solve-field failed with code {result.returncode}: {result.stderr}"
            )

        json_files = list(tmp_path.glob("*.json"))
        if not json_files:
            raise RuntimeError("solve-field did not produce a JSON result.")

        with json_files[0].open() as fh:
            payload = json.load(fh)

        if keep_intermediate:
            # Move files to current working directory for manual inspection.
            output_dir = Path.cwd() / "astrometry_outputs"
            output_dir.mkdir(exist_ok=True)
            for file in tmp_path.iterdir():
                file.rename(output_dir / file.name)

        return {
            "ra": payload["ra"],
            "dec": payload["dec"],
            "orientation": payload["orientation"],
            "pixscale": payload["pixscale"],
            "field_width": payload.get("field_width", float("nan")),
            "field_height": payload.get("field_height", float("nan")),
        }


def _write_fits(image: np.ndarray, path: Path) -> None:
    """Write an image to FITS format without pulling in heavy dependencies."""

    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - dependency check
        raise ImportError(
            "astropy is required to export FITS for solve-field."
        ) from exc

    hdu = fits.PrimaryHDU(data=image)
    hdul = fits.HDUList([hdu])
    hdul.writeto(path, overwrite=True)
