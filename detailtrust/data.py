"""Patient-level NIfTI adapter, deterministic patches, and label-free input transforms."""
import re
import json
import math
import hashlib
from pathlib import Path
import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from .config import MODALITIES, LABEL_MAPPING, write_json, file_hash


def discover_dataset_root(override=None, mount="/kaggle/input"):
    """Use mounted data only. Ambiguous sources require the visible path setting."""
    if override:
        root = Path(override)
        if not root.is_dir():
            raise FileNotFoundError(f"Configured DATASET_ROOT does not exist: {root}")
        return root
    mount = Path(mount)
    candidates = set()
    for path in mount.rglob("BraTS20_Training_*_t1.nii*"):
        relative = path.relative_to(mount)
        depth = 3 if relative.parts[0] == "datasets" else 1
        candidates.add(mount.joinpath(*relative.parts[:depth]))
    if len(candidates) != 1:
        raise ValueError(f"Found {len(candidates)} compatible mounts: {sorted(map(str, candidates))}. Set DATASET_ROOT explicitly; attach the NIfTI mirror first.")
    return next(iter(candidates))


def discover_subjects(root):
    """Group exact subject IDs. Unlabelled challenge validation is inventoried only."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Attach the NIfTI BraTS dataset; missing directory: {root}")
    groups, unmatched = {}, []
    pattern = re.compile(r"^(BraTS20_(?:Training|Validation)_\d+)_(t1ce|t1|t2|flair|seg)\.nii(?:\.gz)?$", re.I)
    for path in sorted(root.rglob("*.nii*")):
        match = pattern.match(path.name)
        if not match:
            unmatched.append(str(path.relative_to(root)))
            continue
        subject, modality = match.groups()
        group = groups.setdefault(subject, {"id": subject, "patient_id": subject, "files": {}})
        if modality.lower() in group["files"]:
            raise ValueError(f"Duplicate {subject}/{modality}; choose a single dataset version/root.")
        group["files"][modality.lower()] = str(path)
    eligible, excluded = [], []
    for subject in sorted(groups):
        record = groups[subject]
        missing = [m for m in MODALITIES if m not in record["files"]]
        if missing or "seg" not in record["files"]:
            excluded.append(dict(id=subject, missing_modalities=missing,
                                 reason="missing modalities" if missing else "no distributed label"))
        elif "_Training_" not in subject:
            excluded.append(dict(id=subject, reason="not in labelled training collection"))
        else:
            eligible.append(record)
    if not eligible:
        raise ValueError("No complete labelled BraTS20 NIfTI subjects found; HDF5/slice mirrors are incompatible.")
    return eligible, dict(discovered=len(groups), eligible=len(eligible), excluded=excluded,
                          unmatched_nifti=unmatched, adapter="BraTS20 exact filenames; no silent rename")


def remap_labels(labels):
    """Original 0/1/2/4 -> exclusive 0/1/2/3; fail on fractional/unknown labels."""
    values = np.unique(labels)
    if not np.isfinite(values).all() or not set(values.tolist()) <= set(LABEL_MAPPING):
        raise ValueError(f"Unexpected BraTS20 label values: {values.tolist()}")
    result = np.zeros(labels.shape, np.uint8)
    for old, new in LABEL_MAPPING.items():
        result[labels == old] = new
    return result


def load_subject(record):
    """Losslessly reorient to RAS; assert aligned orthogonal 1 mm geometry.

    No silent resampling. This adapter deliberately rejects a different spacing
    release; GaussianSplit itself supports physical sigma for other spacings.
    Array axes remain R/A/S and are called D/H/W by PyTorch (no axis transpose).
    """
    import nibabel as nib
    loaded = {name: nib.load(path) for name, path in record["files"].items()}
    original = loaded["t1"]
    canonical = {name: nib.as_closest_canonical(img) for name, img in loaded.items()}
    reference = canonical["t1"]
    spacing = np.asarray(reference.header.get_zooms()[:3], float)
    if not np.allclose(spacing, 1.0, atol=1e-4):
        raise ValueError(f"{record['id']}: spacing {spacing}; configure an explicit resampling protocol.")
    directions = reference.affine[:3, :3] / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-4):
        raise ValueError("Sheared voxel grid is unsupported for physical distance metrics.")
    for name, img in canonical.items():
        if img.shape != reference.shape or not np.allclose(img.affine, reference.affine, atol=1e-4):
            raise ValueError(f"{record['id']}: {name} is not spatially aligned.")
    images = np.stack([canonical[m].get_fdata(dtype=np.float32) for m in MODALITIES])
    if not np.isfinite(images).all():
        raise ValueError(f"{record['id']}: nonfinite MRI intensities")
    labels = remap_labels(canonical["seg"].get_fdata(dtype=np.float32))
    metadata = dict(id=record["id"], spacing=spacing.tolist(), canonical_shape=list(labels.shape),
                    canonical_affine=reference.affine.tolist(), original_shape=list(original.shape),
                    original_affine=original.affine.tolist(), orientation="RAS", array_order="R,A,S",
                    raw_labels=sorted(int(v) for v in np.unique(canonical["seg"].dataobj)))
    return images, labels, metadata


def inventory_geometry(records, output_path):
    """Audit every labelled case; failures stop training instead of being hidden."""
    rows = []
    for record in records:
        try:
            images, labels, metadata = load_subject(record)
            metadata["status"] = "passed"
            metadata["foreground_voxels"] = [int((x != 0).sum()) for x in images]
            metadata["et_volume_mm3"] = float((labels == 3).sum() * np.prod(metadata["spacing"]))
            # Hash image arrays for exact duplicate detection without label-based features.
            metadata["image_sha256"] = hashlib.sha256(images.tobytes()).hexdigest()
            metadata["file_sizes"] = {m: Path(p).stat().st_size for m, p in record["files"].items()}
            rows.append(metadata)
        except (ValueError, OSError) as error:
            rows.append(dict(id=record["id"], status="failed", reason=str(error)))
        write_json(output_path, rows)
    failed = [r for r in rows if r["status"] != "passed"]
    hashes = [r["image_sha256"] for r in rows if r["status"] == "passed"]
    if failed or len(hashes) != len(set(hashes)):
        raise ValueError("Dataset audit failed or duplicate image volumes exist; inspect dataset_integrity.json.")
    return rows


def make_split(records, seed):
    """Custom subject split. At 369 cases use 219/50/100; never call it official."""
    ids = sorted(r["id"] for r in records)
    assert len(ids) == len(set(ids)) and len(ids) >= 6
    shuffled = np.random.default_rng(seed).permutation(ids).tolist()
    if len(ids) == 369:
        n_train, n_val = 219, 50
    else:
        n_train, n_val = max(2, int(len(ids) * 219 / 369)), max(1, int(len(ids) * 50 / 369))
    split = dict(train=shuffled[:n_train], validation=shuffled[n_train:n_train + n_val],
                 test=shuffled[n_train + n_val:], seed=seed,
                 protocol="custom patient split of labelled BraTS20 training collection",
                 official=False, published_author_ids_recovered=False)
    assert_split(split)
    split["sha256"] = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    return split


def assert_split(split):
    parts = [set(split[k]) for k in ("train", "validation", "test")]
    assert all(parts), "Each split must contain at least one subject."
    assert all(len(split[k]) == len(set(split[k])) for k in ("train", "validation", "test"))
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])


def normalize_foreground(images):
    """Per-image nonzero foreground z-score; input features never use labels."""
    result = np.zeros_like(images, dtype=np.float32)
    for modality, image in enumerate(images):
        mask = image != 0
        values = image[mask]
        if not len(values) or values.std() < 1e-6:
            raise ValueError(f"Empty or nearly constant modality {MODALITIES[modality]}")
        result[modality, mask] = (values - values.mean()) / values.std()
    return result


def image_crop(images, available, margin=8):
    """Determine crop solely from available MRI foreground, including at evaluation."""
    foreground = np.any(images[np.asarray(available, bool)] != 0, axis=0)
    if not foreground.any():
        raise ValueError("No image foreground")
    coordinates = np.where(foreground)
    bounds = [(max(0, int(c.min()) - margin), min(n, int(c.max()) + margin + 1))
              for c, n in zip(coordinates, foreground.shape)]
    return tuple(slice(a, b) for a, b in bounds)


def prepare_subject(images, labels, metadata):
    """Cache normalized full volumes; condition-specific cropping happens later."""
    normalized = normalize_foreground(images)
    assert normalized.shape[1:] == labels.shape and labels.dtype == np.uint8
    return normalized, labels, metadata


class SubjectStore:
    """Bounded disk cache with mmap reads; no full dataset in RAM or GPU memory."""
    def __init__(self, records, cache_dir, max_gb=4.0):
        self.records = {r["id"]: r for r in records}
        self.directory = Path(cache_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_gb * 1e9)

    def load(self, subject):
        record = self.records[subject]
        signature = [(m, str(p), Path(p).stat().st_size, Path(p).stat().st_mtime_ns)
                     for m, p in sorted(record["files"].items())]
        key = hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:16]
        stem = self.directory / (subject + "_" + key)
        xp, yp, mp = Path(str(stem) + "_x.npy"), Path(str(stem) + "_y.npy"), Path(str(stem) + ".json")
        if xp.exists() and yp.exists() and mp.exists():
            return np.load(xp, mmap_mode="r"), np.load(yp, mmap_mode="r"), json.loads(mp.read_text())
        images, labels, metadata = prepare_subject(*load_subject(record))
        used = sum(p.stat().st_size for p in self.directory.glob("*.*"))
        if used + images.nbytes + labels.nbytes < self.max_bytes:
            np.save(xp, images)
            np.save(yp, labels)
            write_json(mp, metadata)
        return images, labels, metadata


def sample_availability(rng):
    mask = 15 if rng.random() < 0.25 else int(rng.integers(1, 15))
    return np.asarray([(mask >> i) & 1 for i in range(4)], np.float32)


def extract_patch(images, labels, available, patch, rng, halo=4):
    """Half class-balanced tumour centres, half available-image foreground."""
    classes = [k for k in (1, 2, 3) if np.any(labels == k)]
    if classes and rng.random() < 0.5:
        candidates = np.argwhere(labels == rng.choice(classes))
    else:
        candidates = np.argwhere(np.any(images[available.astype(bool)] != 0, axis=0))
    centre = candidates[rng.integers(len(candidates))]
    starts = [int(c) - n // 2 for c, n in zip(centre, patch)]
    # Reflect at volume borders just as the Gaussian uses reflection.
    pads = [(max(0, halo - s), max(0, s + n + halo - dim))
            for s, n, dim in zip(starts, patch, labels.shape)]
    padded = np.pad(np.asarray(images), [(0, 0)] + pads, mode="reflect")
    ypad = np.pad(np.asarray(labels), pads, mode="constant")
    slices = tuple(slice(s + p[0] - halo, s + p[0] + n + halo)
                   for s, p, n in zip(starts, pads, patch))
    x = padded[(slice(None),) + slices].copy()
    y = ypad[slices][halo:-halo, halo:-halo, halo:-halo].copy()
    return x, y


def corrupt_image(image, family, severity, rng, local=True, axis=2):
    """Image-space simulations. Mask generation never receives a segmentation."""
    image = np.asarray(image, np.float32)
    coarse_shape = None
    if family == "resolution":
        sigma = [0., 0., 0.]
        sigma[axis] = 0.5 * severity
        blurred = ndimage.gaussian_filter(image, sigma, mode="reflect")
        coarse_shape = list(image.shape)
        coarse_shape[axis] = max(2, int(round(image.shape[axis] / severity)))
        t = torch.from_numpy(blurred.copy())[None, None]
        # Adaptive averages define the exact positive row-normalized A_s for A7.
        coarse = F.adaptive_avg_pool3d(t, coarse_shape)
        changed = F.interpolate(coarse, image.shape, mode="trilinear", align_corners=False)[0, 0].numpy()
    elif family == "noise":
        changed = image + rng.normal(0, severity, image.shape).astype(np.float32)
    elif family == "bias":
        grid = rng.normal(0, severity, (4, 4, 4)).astype(np.float32)
        field = F.interpolate(torch.from_numpy(grid)[None, None], image.shape, mode="trilinear", align_corners=False)[0, 0].numpy()
        changed = image * np.clip(np.exp(field), 0.8, 1.2)
    elif family == "ghost":
        changed = image + severity * np.roll(image, image.shape[axis] // 4, axis=axis)
    elif family == "ringing":
        spectrum = np.fft.fft(image, axis=axis)
        frequency = np.abs(np.fft.fftfreq(image.shape[axis]))
        shape = [1, 1, 1]
        shape[axis] = -1
        spectrum *= (frequency <= (0.5 * (1 - severity))).reshape(shape)
        changed = np.fft.ifft(spectrum, axis=axis).real.astype(np.float32)
    else:
        raise ValueError(f"Unknown corruption {family}")
    if local:
        axes = np.ogrid[tuple(slice(0, n) for n in image.shape)]
        centre = [rng.uniform(0.2, 0.8) * n for n in image.shape]
        distance = sum(((a - c) / (0.22 * n)) ** 2 for a, c, n in zip(axes, centre, image.shape))
        mask = np.exp(-0.5 * distance).astype(np.float32)
        changed = image * (1 - mask) + changed * mask
    changed[image == 0] = 0
    return changed.astype(np.float32), coarse_shape


def training_augmentation(images, labels, available, rng, corruption_probability):
    """Aligned flips are interpolation-free; intensity damage affects one scan."""
    images, labels = images.copy(), labels.copy()
    for axis in range(3):
        if rng.random() < 0.5:
            images = np.flip(images, axis + 1).copy()
            labels = np.flip(labels, axis).copy()
    resolution = np.zeros(3, dtype=np.int64)
    if rng.random() < corruption_probability:
        modality = int(rng.choice(np.flatnonzero(available)))
        family = str(rng.choice(["resolution", "noise", "bias"]))
        severity = rng.uniform(1.5, 3) if family == "resolution" else rng.uniform(0.05, 0.15)
        axis = int(rng.integers(3))
        local = bool(rng.integers(2))
        images[modality], coarse = corrupt_image(images[modality], family, severity, rng, local, axis)
        # A7 applies only to global resolution damage; local mixtures do not share one A_s.
        if coarse and not local:
            resolution = np.asarray(labels.shape, np.int64)
            resolution[axis] = max(2, int(round(labels.shape[axis] / severity)))
    return images * available[:, None, None, None], labels, resolution


class PatchDataset(Dataset):
    """Index-deterministic patch stream; resume by optimizer update, no RNG drift."""
    def __init__(self, store, subjects, patch, seed, length, corruption_probability=0.5, start=0):
        self.store, self.subjects, self.patch = store, list(subjects), tuple(patch)
        self.seed, self.length, self.corruption_probability, self.start = seed, length, corruption_probability, start

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.start + index]))
        subject = self.subjects[int(rng.integers(len(self.subjects)))]
        images, labels, _ = self.store.load(subject)
        available = sample_availability(rng)
        images, labels = extract_patch(images, labels, available, self.patch, rng)
        images, labels, resolution = training_augmentation(images, labels, available, rng, self.corruption_probability)
        return dict(images=torch.from_numpy(images), labels=torch.from_numpy(labels.astype(np.int64)),
                    available=torch.from_numpy(available), resolution=torch.from_numpy(resolution), subject=subject)


def make_loader(dataset):
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                      pin_memory=torch.cuda.is_available())


class SyntheticStore:
    """Deterministic tiny blobs solely for execution tests, never a BraTS substitute."""
    def __init__(self, count=12, size=24):
        self.records = {f"synthetic_{i:03}": {} for i in range(count)}
        self.size = size

    def load(self, subject):
        rng = np.random.default_rng(int(subject.split("_")[-1]))
        grid = np.indices((self.size,) * 3)
        distance = sum((g - self.size / 2) ** 2 for g in grid)
        brain = distance < (self.size * 0.44) ** 2
        labels = np.zeros((self.size,) * 3, np.uint8)
        labels[distance < (self.size * 0.24) ** 2] = 2
        labels[distance < (self.size * 0.16) ** 2] = 1
        labels[(grid[0] > self.size / 2) & (distance < (self.size * 0.16) ** 2)] = 3
        images = np.stack([(rng.normal(2, .2, labels.shape) + labels * (m + 1) * .2) * brain for m in range(4)]).astype(np.float32)
        metadata = dict(id=subject, spacing=[1., 1., 1.], canonical_shape=list(labels.shape),
                        canonical_affine=np.eye(4).tolist(), original_shape=list(labels.shape),
                        original_affine=np.eye(4).tolist(), orientation="RAS", synthetic=True)
        return normalize_foreground(images), labels, metadata
