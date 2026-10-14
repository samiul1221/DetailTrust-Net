"""Explicit configurations and experiment definitions (no credentials)."""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import hashlib
import json
import math
import os
import random
import platform
import time
import shutil
import importlib.metadata
import numpy as np
import torch

MODALITIES = ("t1", "t1ce", "t2", "flair")
REGIONS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}
CLASSES = {"background": (0,), "NCR_NET": (1,), "edema": (2,), "ET_class": (3,)}
LABEL_MAPPING = {0: 0, 1: 1, 2: 2, 4: 3}
DATASET_SLUG = "awsaf49/brats20-dataset-training-validation"
WIDTHS = (32, 64, 128, 256, 320)


@dataclass
class RunConfig:
    """An epoch is a reporting interval of sampled patches, not a volume pass."""
    mode: str = "smoke"
    data_root: str = "/kaggle/input/brats20-dataset-training-validation"
    output_dir: str = "/kaggle/working/detailtrust"
    dataset_version: str | None = None
    seed: int = 1729
    seeds: tuple = (1729,)
    split_seed: int = 20261012
    patch: tuple = (96, 96, 96)
    batch_size: int = 1
    train_subjects: int = 4
    val_subjects: int | None = 1
    updates_a: int = 2
    updates_b: int = 2
    updates_c: int = 2
    epoch_updates: int = 2
    validation_every: int = 2
    precision: str = "amp_fp16"
    checkpoint_blocks: bool = True
    workers: int = 0
    lr: float = 2e-4
    gate_lr: float = 1e-3
    weight_decay: float = 1e-5
    warmup_updates: int = 100
    branch_dropout: float = 0.25
    utility_tau: float = 0.1
    utility_weight: float = 0.1
    corruption_probability: float = 0.5
    checkpoint_seconds: float = 1200
    max_hours: float = 1.5
    training_budget_fraction: float = 0.9
    profile_warmup: int = 1
    profile_updates: int = 2
    inference_repeats: int = 2
    early_stopping_patience: int | None = None
    surface_tolerance_mm: float = 1.0
    small_et_threshold_mm3: float | None = None
    nonzero_threshold: float = 1e-5
    variants: tuple = ("B5",)
    run_final_test: bool = False
    protocol_locked: bool = False
    robustness: str = "smoke"
    cache_gb: float = 4.0
    synthetic: bool = False
    resume_dir: str | None = None


def configuration(mode: str = "smoke", **overrides) -> RunConfig:
    """Keep widths fixed in every mode; synthetic smoke alone uses 16³ patches."""
    presets = {
        "smoke": {},
        "pilot": dict(train_subjects=100, val_subjects=5, updates_a=400,
                      updates_b=450, updates_c=150, epoch_updates=50,
                      validation_every=100, profile_warmup=5, profile_updates=100,
                      max_hours=9, robustness="development", cache_gb=8),
        "main": dict(train_subjects=219, val_subjects=None, updates_a=4000,
                     updates_b=4500, updates_c=1500, epoch_updates=100,
                     validation_every=500, profile_warmup=5, profile_updates=100,
                     max_hours=9, robustness="complete", cache_gb=8),
        "extended": dict(train_subjects=219, val_subjects=None, updates_a=4000,
                         updates_b=4500, updates_c=1500, epoch_updates=100,
                         validation_every=500, profile_warmup=5, profile_updates=100,
                         max_hours=9, robustness="complete", cache_gb=8,
                         training_budget_fraction=0.7,
                         seeds=(1729, 1730, 1731),
                         variants=("B0", "B1", "B2", "B4", "B5", "B3", "A1",
                                   "A2", "A3", "A4", "A5", "A6", "A7")),
    }
    if mode not in presets:
        raise ValueError(f"Unknown mode {mode}")
    cfg = RunConfig(**(dict(mode=mode) | presets[mode] | overrides))
    assert cfg.batch_size == 1, "This audited pipeline specifies batch size one."
    assert all(p >= 16 and p % 16 == 0 for p in cfg.patch)
    assert cfg.workers == 0, "Zero workers preserves exact patch-stream resume."
    assert 0 < cfg.training_budget_fraction < 1
    if cfg.run_final_test:
        assert cfg.protocol_locked and mode in ("main", "extended")
        assert not cfg.synthetic
    return cfg


def experiment_registry() -> dict:
    """All requested variants exist even when not selected for this allocation."""
    base = dict(split="learned", anchored=True, gate="spatial", utility=True,
                shuffle=False, joint=False, resolution_loss=False, corruption=True)
    variants = {
        "B0": ("Raw U-Net with modality dropout", dict(gate="raw", corruption=False)),
        "B1": ("Raw U-Net plus matched corruption", dict(gate="raw")),
        "B2": ("Anchored corrections, gates one", dict(gate="ones", utility=False)),
        "B3": ("Four global learned sigmoid scalars", dict(gate="global", utility=False)),
        "B4": ("Spatial gates, segmentation supervision only", dict(utility=False)),
        "B5": ("Full frozen-stage utility-supervised model", {}),
        "A1": ("Spatially permuted utility targets within each branch", dict(shuffle=True)),
        "A2": ("Fixed 0.8 mm Gaussian", dict(split="fixed")),
        "A3": ("Raw context and raw anchored refinement", dict(split="raw")),
        "A4": ("Haar low-pass reconstruction and residual", dict(split="haar")),
        "A5": ("Unanchored correction R(F,H,e)", dict(anchored=False)),
        "A6": ("Unfreeze all components in Stage C", dict(joint=True)),
        "A7": ("Known-resolution occupancy loss, weight 0.1", dict(resolution_loss=True)),
    }
    return {k: dict(description=d, **(base | v)) for k, (d, v) in variants.items()}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def safe_json(value):
    """Undefined metrics become null; their reason belongs in the metric record."""
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, torch.Tensor):
        return safe_json(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value) -> None:
    """Atomic replacement avoids half-written reports after interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(safe_json(value), indent=2, sort_keys=True,
                              allow_nan=False), encoding="utf-8")
    json.loads(tmp.read_text(encoding="utf-8"))
    tmp.replace(path)


def file_hash(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_record(output_dir) -> dict:
    """Whitelist runtime information; never capture environment variables."""
    versions = {}
    for name in ("torch", "numpy", "scipy", "nibabel", "matplotlib", "nbformat"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=platform.python_version(), platform=platform.platform(),
                packages=versions, cuda=torch.version.cuda,
                gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                gpu_count=torch.cuda.device_count(),
                gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory
                if torch.cuda.is_available() else None,
                free_disk_bytes=shutil.disk_usage(output_dir).free)
