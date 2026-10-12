"""Sliding-window score accounting and shared evaluation for every variant."""
import itertools
import hashlib
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from .config import MODALITIES, REGIONS
from .data import image_crop, corrupt_image
from .losses import segmentation_loss, utility_targets
from .metrics import subject_metrics, stability_bound, verify_stability_enumeration


def synchronized_time(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def autocast_context(device, precision):
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=device.type == "cuda" and precision == "amp_fp16")


def window_starts(length, patch):
    stride = max(1, patch // 2)
    starts = list(range(0, max(1, length - patch + 1), stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


@torch.no_grad()
def sliding_window(model, images, available, cfg, device, stage="C", analysis=False):
    """Blend base and each c with the SAME constant overlap weights on CPU.

    No softmax blending or nonlinear postprocessing. Filtering receives a real
    four-voxel halo. Diagnostics q and r are window averages; q_bar*r_bar does
    NOT equal c_bar and must never replace the accumulated contributions.
    """
    model.eval()
    slices = image_crop(images, available)
    crop = np.asarray(images[(slice(None),) + slices])
    original_crop_shape = crop.shape[1:]
    extra = [(0, max(0, p - n)) for n, p in zip(original_crop_shape, cfg.patch)]
    crop = np.pad(crop, [(0, 0)] + extra, mode="constant")
    padded = np.pad(crop, [(0, 0), (4, 4), (4, 4), (4, 4)], mode="reflect")
    shape = crop.shape[1:]
    accum = {"base": torch.zeros(4, *shape), "contributions": torch.zeros(4, 4, *shape)}
    if analysis:
        accum.update(gates=torch.zeros(4, 1, *shape), corrections=torch.zeros(4, 3, *shape),
                     detail_energy=torch.zeros(4, 1, *shape))
    count = torch.zeros(shape)
    flags = torch.as_tensor(available, device=device)[None]
    windows = 0
    started = synchronized_time(device)
    for origin in itertools.product(*(window_starts(n, p) for n, p in zip(shape, cfg.patch))):
        target = tuple(slice(s, s + p) for s, p in zip(origin, cfg.patch))
        source = tuple(slice(s, s + p + 8) for s, p in zip(origin, cfg.patch))
        patch = torch.from_numpy(padded[(slice(None),) + source].copy())[None].to(device)
        with autocast_context(device, cfg.precision):
            output = model(patch, flags, stage=stage)
        for name in accum:
            value = output["detail"][0, :, None].abs() if name == "detail_energy" else output[name][0]
            accum[name][(...,) + target] += value.float().cpu()
        count[target] += 1
        windows += 1
    elapsed = synchronized_time(device) - started
    assert bool((count > 0).all())
    result = {}
    unpad = tuple(slice(0, n) for n in original_crop_shape)
    for name, value in accum.items():
        value = (value / count)[(...,) + unpad]
        full = torch.zeros(*value.shape[:-3], *images.shape[1:])
        full[(...,) + slices] = value
        result[name] = full
    result["logits"] = result["base"] + result["contributions"].sum(0)
    result["runtime"] = dict(seconds=elapsed, windows=windows, patch=list(cfg.patch),
                             full_shape=list(images.shape[1:]), warmup=False)
    return result


def evaluation_conditions(cfg, final=False):
    """Freeze this suite before testing. Ghost/ringing are held out from training."""
    conditions = []
    for bits in range(1, 16):
        flags = [(bits >> i) & 1 for i in range(4)]
        conditions.append(dict(id=f"clean_m{bits:02}", available=flags, corruption=None))
    families = ("noise",) if cfg.robustness == "smoke" else (("ghost", "ringing") if final else ("noise", "resolution"))
    levels = (1,) if cfg.robustness == "smoke" else (1, 2)
    masks = (15,) if cfg.robustness == "smoke" else (15, 13)  # 13 has T1ce absent.
    for family, level, bits in itertools.product(families, levels, masks):
        flags = [(bits >> i) & 1 for i in range(4)]
        severity = {"noise": .075 * level, "resolution": 1.5 + level,
                    "ghost": .15 * level, "ringing": .2 * level}[family]
        conditions.append(dict(id=f"{family}{level}_m{bits:02}_flair", available=flags,
                               corruption=dict(family=family, severity=severity, modality=3, local=True, axis=2)))
    return conditions


def apply_condition(images, subject, condition, seed):
    """Same seed and damage for all variants, independent of execution order."""
    flags = np.asarray(condition["available"], np.float32)
    result = np.asarray(images).copy()
    corruption = condition["corruption"]
    if corruption:
        digest = hashlib.sha256(f"{subject}/{condition['id']}/{seed}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        modality = corruption["modality"]
        assert flags[modality], "Cannot corrupt an absent modality."
        kwargs = {k: v for k, v in corruption.items() if k != "modality"}
        result[modality], _ = corrupt_image(result[modality], rng=rng, **kwargs)
    result *= flags[:, None, None, None]
    return result, flags


def descriptive_values(values):
    values = np.asarray(values, float)
    if not values.size:
        return dict(count=0, mean=None, median=None, q05=None, q95=None, histogram_counts=[], histogram_edges=[])
    counts, edges = np.histogram(values, bins=20)
    return dict(count=values.size, mean=float(values.mean()), median=float(np.median(values)),
                q05=float(np.quantile(values, .05)), q95=float(np.quantile(values, .95)),
                histogram_counts=counts.tolist(), histogram_edges=edges.tolist())


def safe_spearman(x, y):
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    value = float(spearmanr(x, y).statistic)
    return value if np.isfinite(value) else None


def contribution_analysis(output, truth, images, flags, subject, experiment, condition, split, threshold=1e-5):
    """Utility is measured with held-out labels AFTER inference, never a model input."""
    base, contributions, logits = output["base"], output["contributions"], output["logits"]
    labels = torch.from_numpy(np.asarray(truth).astype(np.int64))
    final_ce = F.cross_entropy(logits[None], labels[None], reduction="none")[0]
    candidate_contributions = F.pad(output["corrections"], (0, 0, 0, 0, 0, 0, 1, 0)) * torch.as_tensor(flags)[:, None, None, None, None]
    candidate_logits = base + candidate_contributions.sum(0)
    candidate_ce = F.cross_entropy(candidate_logits[None], labels[None], reduction="none")[0]
    domain = np.any(images[flags.astype(bool)] != 0, axis=0)
    masks = dict(brain=domain, background=domain & (truth == 0), **{k: np.isin(truth, v) for k, v in REGIONS.items()})
    gate_rows, correction_rows = [], []
    for modality in np.flatnonzero(flags):
        without_ce = F.cross_entropy((logits - contributions[modality])[None], labels[None], reduction="none")[0]
        candidate_without = F.cross_entropy((candidate_logits - candidate_contributions[modality])[None], labels[None], reduction="none")[0]
        delta = (without_ce - final_ce).numpy()
        candidate_delta = (candidate_without - candidate_ce).numpy()
        q = output["gates"][modality, 0].numpy()
        r = output["corrections"][modality].numpy()
        c = contributions[modality, 1:].numpy()
        energy = output["detail_energy"][modality, 0].numpy()
        for region, mask in masks.items():
            common = dict(subject=subject, experiment=experiment, condition=condition, split=split,
                          modality=MODALITIES[modality], region=region, voxel_count=int(mask.sum()))
            # Deterministic stride sampling bounds correlation cost; summaries use all voxels.
            indices = np.flatnonzero(mask)[::max(1, int(mask.sum()) // 20000)]
            gate_rows.append(common | descriptive_values(q[mask]) | dict(
                measured_gated_utility_mean=float(delta[mask].mean()) if mask.any() else None,
                helpful_fraction=float((delta[mask] > 0).mean()) if mask.any() else None,
                gate_utility_spearman=safe_spearman(q.ravel()[indices], delta.ravel()[indices]),
                candidate_utility_spearman=safe_spearman(q.ravel()[indices], candidate_delta.ravel()[indices]),
                energy_utility_spearman=safe_spearman(energy.ravel()[indices], delta.ravel()[indices]),
                interpretation="window-averaged gate versus volume-level CE removal; not calibrated quality"))
            correction_rows.append(common | dict(
                absolute_candidate_mean=float(np.abs(r[:, mask]).mean()) if mask.any() else None,
                absolute_contribution_mean=float(np.abs(c[:, mask]).mean()) if mask.any() else None,
                signed_contribution_mean=c[:, mask].mean(1).tolist() if mask.any() else None,
                nonnegligible_fraction=float((np.abs(c[:, mask]).max(0) > threshold).mean()) if mask.any() else None,
                threshold=threshold, class_order=["NCR_NET", "edema", "ET"]))
    bounds, stable = stability_bound(base, contributions, threshold)
    pred, initial = logits.argmax(0).numpy(), base.argmax(0).numpy()
    stable = stable.numpy()
    stability = []
    for region, mask in masks.items():
        n = int(mask.sum())
        stable_mask = mask & stable
        base_error = mask & (initial != truth)
        stability.append(dict(subject=subject, experiment=experiment, condition=condition, split=split,
                              region=region, voxels=n, stable_fraction=float(stable[mask].mean()) if n else None,
                              stable_accuracy=float((pred[stable_mask] == truth[stable_mask]).mean()) if stable_mask.any() else None,
                              repaired_base_errors=int((base_error & (pred == truth)).sum()),
                              base_errors=int(base_error.sum()), introduced_errors=int((mask & (initial == truth) & (pred != truth)).sum()),
                              stable_wrong=int((stable_mask & (pred != truth)).sum()),
                              stable_false_positive=int((stable_mask & (truth == 0) & (pred > 0)).sum()),
                              stable_false_negative=int((stable_mask & (truth > 0) & (pred == 0)).sum())))
    indices = np.flatnonzero(domain)[::max(1, int(domain.sum()) // 2048)]
    check = verify_stability_enumeration(base.reshape(4, -1)[:, indices], contributions.reshape(4, 4, -1)[:, :, indices])
    return gate_rows, correction_rows, stability, check


def evaluate_condition(model, store, subjects, split_manifest, split, condition, experiment, cfg, device, stage="C", analysis=False):
    """Evaluate only an explicitly permitted held-out subject list."""
    assert split in ("validation", "test")
    assert set(subjects) <= set(split_manifest[split])
    assert not set(subjects) & set(split_manifest["train"])
    if split == "test":
        assert cfg.run_final_test and cfg.protocol_locked
    rows, gates, corrections, stability, checks, timings, losses = [], [], [], [], [], [], []
    for subject in subjects:
        images, truth, metadata = store.load(subject)
        damaged, flags = apply_condition(images, subject, condition, cfg.split_seed)
        output = sliding_window(model, damaged, flags, cfg, device, stage, analysis)
        prediction = output["logits"].argmax(0).numpy()
        rows += subject_metrics(prediction, truth, metadata["spacing"], subject, experiment, condition["id"], split, cfg.surface_tolerance_mm)
        loss = segmentation_loss(output["logits"][None], torch.from_numpy(np.asarray(truth).astype(np.int64))[None])
        losses.append(float(loss))
        timings.append(dict(subject=subject, experiment=experiment, condition=condition["id"], split=split, **output["runtime"]))
        if analysis:
            g, c, s, check = contribution_analysis(output, truth, damaged, flags, subject, experiment, condition["id"], split, cfg.nonzero_threshold)
            gates += g; corrections += c; stability += s
            checks.append(dict(subject=subject, experiment=experiment, **check))
            for modality in range(4):
                if flags[modality]:
                    removed = (output["logits"] - output["contributions"][modality]).argmax(0).numpy()
                    rows += subject_metrics(removed, truth, metadata["spacing"], subject, experiment,
                                            condition["id"] + f"_remove_detail_{MODALITIES[modality]}", split, cfg.surface_tolerance_mm)
            rows += subject_metrics(output["base"].argmax(0).numpy(), truth, metadata["spacing"], subject, experiment,
                                    condition["id"] + "_base_only", split, cfg.surface_tolerance_mm)
    return dict(rows=rows, gates=gates, corrections=corrections, stability=stability,
                stability_checks=checks, timings=timings, mean_loss=float(np.mean(losses)))


def validation_score(model, store, subjects, split, cfg, device, stage):
    clean = dict(id="clean_m15", available=[1, 1, 1, 1], corruption=None)
    result = evaluate_condition(model, store, subjects, split, "validation", clean, "selection", cfg, device, stage)
    score = np.mean([r["dice"] for r in result["rows"] if r["region"] in REGIONS])
    return float(score), result["mean_loss"]
