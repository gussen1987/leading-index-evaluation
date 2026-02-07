"""I/O utilities for reading and writing data files."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from risk_index.core.constants import (
    CACHE_DIR,
    PROCESSED_DIR,
    ARTIFACTS_DIR,
    EXPORTS_DIR,
    CONFIG_DIR,
)


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if necessary."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet file."""
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to parquet file."""
    ensure_dir(path.parent)
    df.to_parquet(path, engine="pyarrow")


def read_yaml(path: Path) -> dict:
    """Read a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(data: dict, path: Path) -> None:
    """Write data to YAML file."""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def read_json(path: Path) -> dict:
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: dict, path: Path, indent: int = 2) -> None:
    """Write data to JSON file."""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def write_excel(
    sheets: dict[str, pd.DataFrame],
    path: Path,
    index: bool = True,
) -> None:
    """Write multiple DataFrames to Excel workbook.

    Args:
        sheets: Dict mapping sheet names to DataFrames
        path: Output file path
        index: Whether to include DataFrame index
    """
    ensure_dir(path.parent)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=index)


def get_cache_path(source: str, series_id: str) -> Path:
    """Get cache file path for a series."""
    return CACHE_DIR / f"{source}_{series_id}.parquet"


def get_processed_path(name: str) -> Path:
    """Get processed data file path."""
    return PROCESSED_DIR / f"{name}.parquet"


def get_artifact_path(name: str, ext: str = "json") -> Path:
    """Get artifact file path."""
    return ARTIFACTS_DIR / f"{name}.{ext}"


def get_export_path(name: str) -> Path:
    """Get export file path."""
    return EXPORTS_DIR / f"{name}.xlsx"


def is_cache_stale(path: Path, max_age_days: int = 1) -> bool:
    """Check if cache file is older than max age."""
    if not path.exists():
        return True
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    age = datetime.now() - mtime
    return age.days >= max_age_days


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()[:16]}"


def hash_config_files() -> dict[str, str]:
    """Hash all configuration files for reproducibility tracking."""
    config_files = [
        "sources.yml",
        "universe.yml",
        "transforms.yml",
        "composites.yml",
        "regimes.yml",
        "checklist.yml",
        "backtest.yml",
    ]
    hashes = {}
    for filename in config_files:
        path = CONFIG_DIR / filename
        if path.exists():
            hashes[filename] = hash_file(path)
    return hashes


def save_run_manifest(
    data_as_of: str,
    universe_size: dict[str, int],
    selected_features: dict[str, int],
    git_commit: str | None = None,
) -> Path:
    """Save run manifest for reproducibility.

    Args:
        data_as_of: Last data date
        universe_size: Dict with series/ratios/features counts
        selected_features: Dict with feature counts per speed
        git_commit: Optional git commit hash

    Returns:
        Path to saved manifest
    """
    manifest = {
        "run_timestamp": datetime.now().isoformat(),
        "git_commit": git_commit,
        "config_hashes": hash_config_files(),
        "data_as_of": data_as_of,
        "universe_size": universe_size,
        "selected_features": selected_features,
    }

    path = get_artifact_path("run_manifest", "json")
    write_json(manifest, path)
    return path


def load_latest_or_dated(
    base_name: str,
    directory: Path,
    ext: str = "parquet",
    use_latest: bool = True,
) -> pd.DataFrame:
    """Load latest file or dated version.

    Args:
        base_name: Base filename without extension
        directory: Directory to search
        ext: File extension
        use_latest: If True, use *_latest file; else find most recent dated

    Returns:
        Loaded DataFrame
    """
    if use_latest:
        path = directory / f"{base_name}_latest.{ext}"
        if path.exists():
            return read_parquet(path) if ext == "parquet" else pd.read_excel(path)

    # Find most recent dated file
    pattern = f"{base_name}_*.{ext}"
    files = sorted(directory.glob(pattern), reverse=True)
    if files:
        path = files[0]
        return read_parquet(path) if ext == "parquet" else pd.read_excel(path)

    raise FileNotFoundError(f"No files found matching {pattern} in {directory}")


def save_with_latest(
    df: pd.DataFrame,
    base_name: str,
    directory: Path,
    archive: bool = False,
    ext: str = "parquet",
) -> Path:
    """Save file with _latest suffix, optionally archive dated copy.

    Args:
        df: DataFrame to save
        base_name: Base filename
        directory: Output directory
        archive: If True, also save dated copy
        ext: File extension

    Returns:
        Path to latest file
    """
    ensure_dir(directory)

    # Save latest
    latest_path = directory / f"{base_name}_latest.{ext}"
    if ext == "parquet":
        write_parquet(df, latest_path)
    else:
        df.to_excel(latest_path, index=True)

    # Optionally archive
    if archive:
        date_str = datetime.now().strftime("%Y%m%d")
        archive_path = directory / f"{base_name}_{date_str}.{ext}"
        if ext == "parquet":
            write_parquet(df, archive_path)
        else:
            df.to_excel(archive_path, index=True)

    return latest_path
