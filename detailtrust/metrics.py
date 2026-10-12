"""Patient-level overlap, physical surface distances, lesion metrics, and stability."""
import itertools
import numpy as np
from scipy import ndimage
from scipy.stats import spearmanr
import torch
from .config import REGIONS, CLASSES


def binary_metrics(prediction, truth, spacing, tolerance_mm=1.0):
    """Macro aggregation is performed later; all empty-case decisions are explicit."""
    prediction, truth = prediction.astype(bool), truth.astype(bool)
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    tn = int((~prediction & ~truth).sum())
    p, t = tp + fp, tp + fn
    both_empty, one_empty = p == 0 and t == 0, (p == 0) != (t == 0)
    result = dict(dice=2 * tp / (p + t) if p + t else 1.,
                  iou=tp / (tp + fp + fn) if tp + fp + fn else 1.,
                  precision=tp / p if p else None, recall=tp / t if t else None,
                  specificity=tn / (tn + fp) if tn + fp else None,
                  tp=tp, fp=fp, fn=fn, tn=tn, truth_voxels=t, prediction_voxels=p,
                  both_empty=both_empty, one_empty=one_empty,
                  fp_mm3=float(fp * np.prod(spacing)),
                  absolute_volume_error_mm3=float(abs(p - t) * np.prod(spacing)),
                  undefined_policy="precision null if prediction empty; recall null if truth empty; HD95 null if exactly one empty")
    if both_empty or one_empty:
        result.update(hd95_mm=0. if both_empty else None,
                      surface_dice=1. if both_empty else 0.)
        return result
    structure = ndimage.generate_binary_structure(3, 1)
    ps = prediction & ~ndimage.binary_erosion(prediction, structure=structure, border_value=0)
    ts = truth & ~ndimage.binary_erosion(truth, structure=structure, border_value=0)
    p_to_t = ndimage.distance_transform_edt(~ts, sampling=spacing)[ps]
    t_to_p = ndimage.distance_transform_edt(~ps, sampling=spacing)[ts]
    result["hd95_mm"] = float(np.percentile(np.concatenate((p_to_t, t_to_p)), 95))
    result["surface_dice"] = float(((p_to_t <= tolerance_mm).sum() + (t_to_p <= tolerance_mm).sum()) / (len(p_to_t) + len(t_to_p)))
    return result


def lesion_metrics(prediction, truth):
    """26-connected ET lesions; one-to-one maximum-cardinality overlap matching."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    pred, npred = ndimage.label(prediction, structure=np.ones((3, 3, 3)))
    true, ntrue = ndimage.label(truth, structure=np.ones((3, 3, 3)))
    pairs = np.unique(np.stack((pred.ravel(), true.ravel()), 1), axis=0)
    pairs = pairs[(pairs[:, 0] > 0) & (pairs[:, 1] > 0)]
    if len(pairs):
        graph = csr_matrix((np.ones(len(pairs)), (pairs[:, 0]-1, pairs[:, 1]-1)), shape=(npred, ntrue))
        matched = int((maximum_bipartite_matching(graph, perm_type="column") >= 0).sum())
    else:
        matched = 0
    return dict(predicted_lesions=npred, reference_lesions=ntrue, matched_lesions=matched,
                lesion_precision=matched / npred if npred else None,
                lesion_recall=matched / ntrue if ntrue else None)


def subject_metrics(prediction, truth, spacing, subject, experiment, condition, split, tolerance=1.):
    rows = []
    for region, classes in (REGIONS | CLASSES).items():
        row = binary_metrics(np.isin(prediction, classes), np.isin(truth, classes), spacing, tolerance)
        row.update(subject=subject, experiment=experiment, condition=condition, split=split, region=region,
                   metric_scope="region" if region in REGIONS else "exclusive_class",
                   aggregation="per_patient", surface_tolerance_mm=tolerance,
                   surface_convention="6-connected boundary voxel centres; unweighted surface point Dice",
                   hd95_convention="95th percentile of concatenated directed distances")
        if region == "ET":
            row.update(lesion_metrics(prediction == 3, truth == 3))
        rows.append(row)
    return rows


def aggregate_metrics(rows):
    """Separate each experiment/condition/split/region; count undefined cases."""
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("experiment", "condition", "split", "region"))
        groups.setdefault(key, []).append(row)
    results = []
    for key, values in groups.items():
        summary = dict(zip(("experiment", "condition", "split", "region"), key))
        summary.update(subjects=[v["subject"] for v in values], subject_count=len(values), aggregation="patient_macro")
        for metric in ("dice", "iou", "precision", "recall", "specificity", "hd95_mm", "surface_dice", "fp_mm3", "absolute_volume_error_mm3", "lesion_precision", "lesion_recall"):
            data = np.asarray([v[metric] for v in values if v.get(metric) is not None], float)
            summary[metric] = dict(mean=float(data.mean()) if len(data) else None,
                                   std=float(data.std(ddof=1)) if len(data) > 1 else None,
                                   median=float(np.median(data)) if len(data) else None,
                                   q05=float(np.quantile(data, .05)) if len(data) else None,
                                   q95=float(np.quantile(data, .95)) if len(data) else None,
                                   valid_count=len(data), undefined_count=len(values)-len(data))
        summary["both_empty_count"] = sum(v["both_empty"] for v in values)
        summary["one_empty_count"] = sum(v["one_empty"] for v in values)
        results.append(summary)
    return results


def stability_bound(base, contributions, tolerance=1e-5):
    """base [4,...], c [4 modalities,4 classes,...], fixed-context attenuation."""
    logits = base + contributions.sum(0)
    winner = logits.argmax(0)
    winning_logit = torch.gather(logits, 0, winner[None])[0]
    winning_contributions = torch.gather(contributions, 1, winner[None, None].expand(contributions.shape[0], 1, *winner.shape))
    difference = winning_contributions - contributions
    bounds = winning_logit[None] - logits - difference.clamp_min(0).sum(0)
    bounds.scatter_(0, winner[None], float("inf"))
    minimum = bounds.min(0).values
    return minimum, minimum > tolerance


def verify_stability_enumeration(base, contributions, tolerance=1e-5):
    """Compare analytic rival margins against all 16 binary retention subsets."""
    logits = base + contributions.sum(0)
    winner = logits.argmax(0)
    worst = torch.full_like(logits, float("inf"))
    for bits in itertools.product((0., 1.), repeat=4):
        weights = contributions.new_tensor(bits).view(4, *([1] * (contributions.ndim - 1)))
        scores = base + (weights * contributions).sum(0)
        margins = torch.gather(scores, 0, winner[None]) - scores
        margins.scatter_(0, winner[None], float("inf"))
        worst = torch.minimum(worst, margins)
    bound, stable = stability_bound(base, contributions, tolerance)
    error = float((bound - worst.min(0).values).abs().max())
    return dict(max_margin_error=error, passed=error <= tolerance,
                subset_count=16, voxel_count=winner.numel())


def paired_patient_comparison(rows, candidate, baseline, bootstrap_seed=42):
    """Resample patients after averaging their matched region/condition scores."""
    scores = {}
    for row in rows:
        if row["experiment"] in (candidate, baseline) and row["region"] in REGIONS:
            key = (row["subject"], row["condition"], row["region"], row["split"])
            scores.setdefault(key, {})[row["experiment"]] = row["dice"]
    patient_values = {}
    for key, pair in scores.items():
        if candidate in pair and baseline in pair:
            patient_values.setdefault(key[0], []).append((pair[candidate], pair[baseline]))
    if not patient_values:
        return dict(status="not_run", reason="No matched patient-condition records")
    pairs = np.asarray([np.mean(values, axis=0) for values in patient_values.values()])
    difference = pairs[:, 0] - pairs[:, 1]
    rng = np.random.default_rng(bootstrap_seed)
    sampled = rng.choice(difference, size=(2000, len(difference)), replace=True).mean(1)
    base_mean = float(pairs[:, 1].mean())
    return dict(status="completed", candidate=candidate, baseline=baseline,
                patients=sorted(patient_values), patient_count=len(difference),
                absolute_dice_difference=float(difference.mean()),
                relative_difference=float(difference.mean() / base_mean) if base_mean else None,
                patient_bootstrap_ci95=np.quantile(sampled, [.025, .975]).tolist() if len(difference) > 1 else None,
                replicates=2000, unit="patient", warning="Conditional on trained weights; not training-seed uncertainty")
