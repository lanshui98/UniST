"""Locate UniST external backends (InterpolAI, SUICA_pro)."""

from __future__ import annotations

from pathlib import Path


def package_root() -> Path:
    """Root of the UniST package (repo root in editable installs)."""
    return Path(__file__).resolve().parents[1]


def interpolai_dir() -> Path:
    return package_root() / "external" / "InterpolAI"


def interpolai_model_dir() -> Path:
    return interpolai_dir() / "interpolation" / "model"


def suica_dir() -> Path:
    return package_root() / "external" / "SUICA_pro"


def require_interpolai_model() -> Path:
    path = interpolai_model_dir()
    if not path.exists():
        raise FileNotFoundError(
            f"InterpolAI weights not found at {path}.\n"
            "Download from Google Drive and place under "
            "external/InterpolAI/interpolation/model/\n"
            "See README / https://unist-tutorial.readthedocs.io/en/latest/interpolation.html"
        )
    return path


def require_suica() -> Path:
    path = suica_dir()
    if not (path / "systems").exists():
        raise FileNotFoundError(
            f"SUICA_pro not found at {path}. "
            "Install UniST from the full GitHub repository."
        )
    return path
