"""The exact full-width model in arch.md, with explicit named ablations."""
import math
from contextlib import nullcontext
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .config import WIDTHS


def availability_maps(images, available):
    """images [B,4,D,H,W]; available [B,4]. Reject absent evidence."""
    assert images.ndim == 5 and images.shape[1] == 4
    assert available.shape == images.shape[:2]
    assert torch.all((available == 0) | (available == 1))
    assert torch.all(available.sum(1) > 0), "At least one MRI must be present."
    maps = available.to(images).view(*available.shape, 1, 1, 1)
    return images * maps, maps.expand_as(images)


def gaussian_kernel(sigma_mm: float, spacing_mm: float, radius: int, device, dtype):
    """Convert physical sigma to voxels; all taps sum to one."""
    assert sigma_mm > 0 and spacing_mm > 0
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (coordinates / (sigma_mm / spacing_mm)) ** 2)
    return kernel / kernel.sum()


def gaussian_blur(images, sigma_mm, spacing=(1., 1., 1.)):
    """Separable, differentiable, nine-tap filtering on 1 mm data."""
    result = images
    channels = images.shape[1]
    for axis, step in enumerate(spacing):
        radius = max(4, math.ceil(3 * 1.2 / step))
        kernel = gaussian_kernel(sigma_mm, step, radius, images.device, images.dtype)
        shape = [1, 1, 1]
        shape[axis] = kernel.numel()
        weight = kernel.view(1, 1, *shape).expand(channels, 1, *shape)
        padding = [0] * 6
        padding[2 * (2 - axis):2 * (2 - axis) + 2] = [radius, radius]
        assert result.shape[axis + 2] > radius, "Patch is too small for reflection."
        result = F.conv3d(F.pad(result, padding, mode="reflect"), weight, groups=channels)
    return result


class GaussianSplit(nn.Module):
    """Twelve global logits; no subject-dependent smoothing network."""
    def __init__(self, learned=True):
        super().__init__()
        if learned:
            self.alpha = nn.Parameter(torch.tensor([[-1., 1., -1.]]).repeat(4, 1))
        else:
            self.register_parameter("alpha", None)

    def forward(self, images, spacing=(1., 1., 1.)):
        if self.alpha is None:
            low = gaussian_blur(images, 0.8, spacing)
        else:
            weights = self.alpha.softmax(-1)
            low = torch.zeros_like(images)
            for index, sigma in enumerate((0.5, 0.8, 1.2)):
                low = low + gaussian_blur(images, sigma, spacing) * weights[:, index].view(1, 4, 1, 1, 1)
        return low, images - low


class HaarSplit(nn.Module):
    """One-level orthonormal Haar: zero high bands, inverse low band, H=X-L.

    Pairwise averaging followed by repeat is exactly that low reconstruction.
    Even dimensions and an even (four-voxel) halo preserve the pair origin.
    """
    def forward(self, images, spacing=(1., 1., 1.)):
        assert all(n % 2 == 0 for n in images.shape[-3:])
        low = F.avg_pool3d(images, 2)
        for axis in (2, 3, 4):
            low = low.repeat_interleave(2, dim=axis)
        return low, images - low


class ResidualBlock(nn.Module):
    """Two bias-free convolutions, eight-group affine norms, projected skip."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.projection = nn.Conv3d(in_channels, out_channels, 1, bias=False)

    def forward(self, images):
        features = F.leaky_relu(self.norm1(self.conv1(images)), 0.01)
        features = self.norm2(self.conv2(features)) + self.projection(images)
        return F.leaky_relu(features, 0.01)


def run_block(block, features, recompute=False):
    """Non-reentrant checkpointing changes memory use, not the model equation."""
    if recompute and torch.is_grad_enabled():
        return checkpoint(block, features, use_reentrant=False)
    return block(features)


class ResidualUNet(nn.Module):
    """8-channel input -> four logits and a 32-channel full-resolution feature."""
    def __init__(self, checkpoint_blocks=True):
        super().__init__()
        self.checkpoint_blocks = checkpoint_blocks
        self.encoders = nn.ModuleList(ResidualBlock(a, b) for a, b in
                                     zip((8,) + WIDTHS[:-1], WIDTHS))
        self.decoders = nn.ModuleList(ResidualBlock(a + b, b) for a, b in
                                     zip(WIDTHS[:0:-1], WIDTHS[-2::-1]))
        self.head = nn.Conv3d(32, 4, 1, bias=True)

    def forward(self, images):
        assert all(n % 16 == 0 for n in images.shape[-3:])
        skips = []
        features = images
        for index, block in enumerate(self.encoders):
            if index:
                features = F.avg_pool3d(features, 2)
            features = run_block(block, features, self.checkpoint_blocks)
            skips.append(features)
        for block, skip in zip(self.decoders, reversed(skips[:-1])):
            features = F.interpolate(features, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            features = run_block(block, torch.cat((features, skip), 1), self.checkpoint_blocks)
        return self.head(features), features


def modality_identity(reference, modality):
    """A four-channel one-hot map, not a learned embedding."""
    identity = reference.new_zeros((reference.shape[0], 4, 1, 1, 1))
    identity[:, modality] = 1
    return identity.expand(-1, -1, *reference.shape[-3:])


class DetailCorrection(nn.Module):
    """Shared R: 37 -> 16 -> 16 -> 3. No normalization or dropout."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv3d(37, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv3d(16, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv3d(16, 3, 1))
        nn.init.normal_(self.layers[-1].weight, std=0.001)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features, detail, identity, anchored=True):
        correction = self.layers(torch.cat((features, detail, identity), 1))
        if anchored:
            correction = correction - self.layers(torch.cat((features, torch.zeros_like(detail), identity), 1))
        return correction


class SpatialUtilityGate(nn.Module):
    """32+1+3+4+4=44 channels -> one independent sigmoid at each voxel."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv3d(44, 16, 1), nn.ReLU(),
                                    nn.Conv3d(16, 16, 3, padding=1, groups=16), nn.ReLU(),
                                    nn.Conv3d(16, 1, 1), nn.Sigmoid())

    def forward(self, features, detail, correction, identity, flags):
        inputs = torch.cat((features, detail, correction, identity, flags), 1)
        assert inputs.shape[1] == 44
        return self.layers(inputs)


def fuse_logits(base_logits, corrections, gates, available, retention=None):
    """r [B,M,3,D,H,W], q [B,M,1,D,H,W]; return exact foreground sum."""
    weight = available[:, :, None, None, None, None].to(corrections)
    if retention is not None:
        weight = weight * retention[:, :, None, None, None, None]
    foreground = weight * gates * corrections
    background = torch.zeros_like(foreground[:, :, :1])
    contributions = torch.cat((background, foreground), 2).float()
    logits = base_logits.float() + contributions.sum(1)
    return logits, contributions


class DetailTrustNet(nn.Module):
    """arch.md reference model. Ablations are explicit constructor arguments."""
    def __init__(self, split="learned", anchored=True, gate="spatial", checkpoint_blocks=True):
        super().__init__()
        self.architecture_config = dict(split=split, anchored=anchored, gate=gate,
                                        checkpoint_blocks=checkpoint_blocks)
        self.split_kind, self.anchored, self.gate_kind = split, anchored, gate
        self.backbone = ResidualUNet(checkpoint_blocks)
        self.splitter = GaussianSplit(split == "learned") if split in ("learned", "fixed") else HaarSplit() if split == "haar" else None
        self.refiner = DetailCorrection() if gate != "raw" else None
        self.gate = SpatialUtilityGate() if gate == "spatial" else None
        self.global_logits = nn.Parameter(torch.zeros(4)) if gate == "global" else None
        self.freeze_candidates = False

    def candidates(self, images, available, halo=4, compute_detail=True):
        """Filter the halo first; feed only the central patch to the U-Net."""
        images, flags = availability_maps(images, available)
        low, detail = self.splitter(images) if self.splitter else (images, images)
        crop = (slice(None), slice(None)) + (slice(halo, -halo),) * 3 if halo else (...,)
        low, detail, flags = low[crop], detail[crop], flags[crop]
        base, features = self.backbone(torch.cat((low, flags), 1))
        corrections = []
        for modality in range(4):
            if not compute_detail or self.refiner is None or not bool(available[:, modality].any()):
                correction = torch.zeros_like(base[:, 1:])
            else:
                identity = modality_identity(features, modality)
                def refine(f, h, e):
                    return self.refiner(f, h, e, self.anchored)
                args = (features, detail[:, modality:modality + 1], identity)
                correction = checkpoint(refine, *args, use_reentrant=False) if self.backbone.checkpoint_blocks and torch.is_grad_enabled() else refine(*args)
                correction = correction * available[:, modality, None, None, None, None]
            corrections.append(correction)
        return base, features, detail, flags, torch.stack(corrections, 1)

    def forward(self, images, available, stage="C", halo=4, retention=None):
        with torch.no_grad() if self.freeze_candidates else nullcontext():
            base, features, detail, flags, corrections = self.candidates(images, available, halo, stage != "A")
        gates = []
        for modality in range(4):
            if stage in ("A", "B") or self.gate_kind in ("ones", "raw"):
                gate = torch.ones_like(base[:, :1])
            elif self.gate_kind == "global":
                gate = self.global_logits[modality].sigmoid().expand_as(base[:, :1])
            else:
                gate = self.gate(features, detail[:, modality:modality + 1], corrections[:, modality],
                                 modality_identity(features, modality), flags)
            gates.append(gate)
        gates = torch.stack(gates, 1)
        if stage == "A":
            corrections = corrections * 0
        logits, contributions = fuse_logits(base, corrections, gates, available, retention)
        return dict(logits=logits, base=base.float(), corrections=corrections.float(),
                    gates=gates.float(), contributions=contributions,
                    detail=detail, available=available)


def build_model(variant, checkpoint_blocks=True):
    """Raw baselines have no dormant splitter/refiner/gate parameters."""
    raw = variant["gate"] == "raw"
    return DetailTrustNet(split="raw" if raw else variant["split"],
                          anchored=variant["anchored"], gate=variant["gate"],
                          checkpoint_blocks=checkpoint_blocks)


def parameter_counts(model):
    return dict(total=sum(p.numel() for p in model.parameters()),
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                by_child={n: sum(p.numel() for p in m.parameters()) for n, m in model.named_children()})
