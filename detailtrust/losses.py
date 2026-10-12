"""Float32 region losses and the exact detached Stage C utility target."""
import torch
import torch.nn.functional as F
from .model import fuse_logits


def region_probabilities(logits):
    probabilities = logits.float().softmax(1)
    return torch.stack((probabilities[:, 1:].sum(1), probabilities[:, 1] + probabilities[:, 3], probabilities[:, 3]), 1)


def region_labels(labels):
    """WT, TC, ET in that order, derived from exclusive internal labels."""
    return torch.stack((labels > 0, (labels == 1) | (labels == 3), labels == 3), 1)


def segmentation_loss(logits, labels):
    """CE + mean region soft-Dice, reduced per patient then across patients."""
    logits = logits.float()
    truth, probabilities = region_labels(labels).float(), region_probabilities(logits)
    axes = (2, 3, 4)
    intersection = (truth * probabilities).sum(axes)
    dice = (2 * intersection + 1e-5) / (truth.sum(axes) + probabilities.sum(axes) + 1e-5)
    return F.cross_entropy(logits, labels) + (1 - dice).mean()


def utility_targets(base, corrections, available, labels, tau=0.1):
    """t_m = sigmoid(pool(CE(z_all-r_m)-CE(z_all))/tau), completely detached.

    Sum all AVAILABLE ungated candidate corrections. Positive delta means that
    removing this correction increases CE, so the correction helped. Pool only
    spatial axes, using a truncated neighbourhood at volume boundaries.
    """
    with torch.no_grad():
        ones = torch.ones_like(corrections[:, :, :1])
        all_logits, candidate_contributions = fuse_logits(base, corrections, ones, available)
        reference_ce = F.cross_entropy(all_logits.float(), labels, reduction="none")
        targets, deltas = [], []
        for modality in range(4):
            without = all_logits - candidate_contributions[:, modality]
            delta = F.cross_entropy(without, labels, reduction="none") - reference_ce
            smooth = F.avg_pool3d(delta[:, None], 3, stride=1, padding=1, count_include_pad=False)
            targets.append(torch.sigmoid(smooth / tau))
            deltas.append(delta[:, None])
        return torch.stack(targets, 1).detach(), torch.stack(deltas, 1).detach()


def balanced_gate_bce(gates, targets, labels, available):
    """Equal WT/background weighting per subject and available modality."""
    losses = F.binary_cross_entropy(gates.float().clamp(1e-7, 1 - 1e-7), targets.float(), reduction="none")
    terms = []
    for patient in range(labels.shape[0]):
        masks = [labels[patient] > 0, labels[patient] == 0]
        masks = [mask for mask in masks if bool(mask.any())]
        for modality in range(4):
            if bool(available[patient, modality]):
                terms.append(torch.stack([losses[patient, modality, 0][mask].mean() for mask in masks]).mean())
    return torch.stack(terms).mean()


def resolution_occupancy_loss(logits, labels, coarse_shape):
    """Positive row-normalized adaptive averages for a declared coarse lattice."""
    probabilities = logits.float().softmax(1)
    one_hot = F.one_hot(labels, 4).permute(0, 4, 1, 2, 3).float()
    truth = F.adaptive_avg_pool3d(one_hot, coarse_shape)
    prediction = F.adaptive_avg_pool3d(probabilities, coarse_shape)
    return -(truth * prediction.clamp_min(1e-7).log()).sum(1).mean()


def stage_objective(outputs, labels, stage, cfg, variant, resolution=None):
    """Return the scalar loss and separately logged, named components."""
    available, base, corrections = outputs["available"], outputs["base"], outputs["corrections"]
    terms = {}
    if stage in ("A", "RAW"):
        terms["segmentation"] = segmentation_loss(outputs["logits"], labels)
        loss = terms["segmentation"]
    elif stage == "B":
        retention = (torch.rand_like(available) >= cfg.branch_dropout).float()
        logits, _ = fuse_logits(base, corrections, torch.ones_like(outputs["gates"]), available, retention)
        terms["segmentation"] = segmentation_loss(logits, labels)
        terms["base"] = segmentation_loss(base, labels)
        auxiliary = []
        for patient in range(labels.shape[0]):
            for modality in range(4):
                if bool(available[patient, modality]):
                    addition = F.pad(corrections[patient:patient+1, modality], (0, 0, 0, 0, 0, 0, 1, 0))
                    auxiliary.append(segmentation_loss(base[patient:patient+1] + addition, labels[patient:patient+1]))
        terms["branch_auxiliary"] = torch.stack(auxiliary).mean()
        loss = terms["segmentation"] + 0.3 * terms["base"] + 0.1 * terms["branch_auxiliary"]
    elif stage == "C":
        terms["segmentation"] = segmentation_loss(outputs["logits"], labels)
        loss = terms["segmentation"]
        if variant["utility"]:
            target, delta = utility_targets(base, corrections, available, labels, cfg.utility_tau)
            if variant["shuffle"]:
                # Same marginal targets; destroy spatial alignment within each branch.
                flat = target.flatten(3)
                order = torch.randperm(flat.shape[-1], device=flat.device)
                target = flat[..., order].reshape_as(target)
            terms["utility_bce"] = balanced_gate_bce(outputs["gates"], target, labels, available)
            terms["utility_target_mean"] = target.mean()
            loss = loss + cfg.utility_weight * terms["utility_bce"]
    else:
        raise ValueError(stage)
    if variant["resolution_loss"] and resolution is not None and bool((resolution > 0).all()):
        shape = tuple(int(x) for x in resolution[0].tolist())
        terms["resolution_occupancy"] = resolution_occupancy_loss(outputs["logits"], labels, shape)
        loss = loss + 0.1 * terms["resolution_occupancy"]
    terms["total"] = loss
    return loss, {name: float(value.detach()) for name, value in terms.items()}
