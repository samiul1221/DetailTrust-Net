"""Small synthetic tests; passing these does not demonstrate medical performance."""
from dataclasses import replace
from pathlib import Path
import tempfile
import time
import json
import itertools
import importlib.util
import numpy as np
import torch
from .config import configuration, experiment_registry, write_json, seed_everything
from .data import remap_labels, make_split, assert_split, SyntheticStore, PatchDataset, make_loader, image_crop, corrupt_image
from .model import DetailTrustNet, GaussianSplit, HaarSplit, SpatialUtilityGate, DetailCorrection, modality_identity, availability_maps, gaussian_kernel, fuse_logits, parameter_counts
from .losses import segmentation_loss, region_labels, utility_targets, balanced_gate_bce, stage_objective
from .metrics import binary_metrics, stability_bound, verify_stability_enumeration, lesion_metrics
from .evaluation import sliding_window, evaluate_condition
from .training import configure_stage, make_optimizer, train_update, save_checkpoint, load_checkpoint


def check_labels_and_splits():
    assert remap_labels(np.array([0, 1, 2, 4])).tolist() == [0, 1, 2, 3]
    try:
        remap_labels(np.array([3]))
        raise AssertionError("Unknown original labels were accepted")
    except ValueError:
        pass
    labels = torch.tensor([[[[0, 1, 2, 3]]]])
    regions = region_labels(labels)
    assert regions[0, 0].sum() == 3 and regions[0, 1].sum() == 2 and regions[0, 2].sum() == 1
    split = make_split([dict(id=f"p{i}") for i in range(369)], 9)
    assert [len(split[k]) for k in ("train", "validation", "test")] == [219, 50, 100]
    assert_split(split)
    assert split == make_split([dict(id=f"p{i}") for i in range(369)], 9)


def check_gaussian_and_haar():
    x = torch.randn(1, 4, 16, 16, 16)
    for spacing in (1., 2.):
        kernel = gaussian_kernel(.8, spacing, 4, x.device, x.dtype)
        assert torch.allclose(kernel.sum(), torch.tensor(1.))
        assert torch.allclose(kernel, kernel.flip(0))
    narrow = gaussian_kernel(.8, 2., 4, x.device, x.dtype)
    wide = gaussian_kernel(.8, 1., 4, x.device, x.dtype)
    assert narrow[4] > wide[4]
    for split in (GaussianSplit(), GaussianSplit(False), HaarSplit()):
        low, high = split(x)
        torch.testing.assert_close(low + high, x)
    split = GaussianSplit()
    assert split.alpha.shape == (4, 3)
    torch.testing.assert_close(split.alpha[0], torch.tensor([-1., 1., -1.]))
    low, high = split(x)
    low.square().mean().backward()
    assert torch.isfinite(split.alpha.grad).all() and split.alpha.grad.abs().sum() > 0


def check_availability_and_anchor():
    x = torch.randn(1, 4, 16, 16, 16)
    a = torch.tensor([[1., 0., 1., 0.]])
    masked, maps = availability_maps(x, a)
    assert not masked[:, 1].any() and not maps[:, 3].any()
    try:
        availability_maps(x, torch.zeros_like(a))
        raise AssertionError("All-missing case was accepted")
    except AssertionError as error:
        assert "At least one" in str(error)
    features = torch.randn(1, 32, 8, 8, 8)
    identity = modality_identity(features, 2)
    detail = torch.zeros(1, 1, 8, 8, 8)
    correction = DetailCorrection()(features, detail, identity)
    assert torch.count_nonzero(correction) == 0
    gate = SpatialUtilityGate()(features, detail, correction, identity, torch.ones(1, 4, 8, 8, 8))
    assert gate.shape == (1, 1, 8, 8, 8) and bool(((gate >= 0) & (gate <= 1)).all())


def check_fusion_and_stability():
    base = torch.randn(1, 4, 4, 4, 4)
    corrections = torch.randn(1, 4, 3, 4, 4, 4)
    gates = torch.rand(1, 4, 1, 4, 4, 4)
    available = torch.tensor([[1., 0., 1., 1.]])
    logits, contributions = fuse_logits(base, corrections, gates, available)
    torch.testing.assert_close(logits[:, 0], base[:, 0])
    assert torch.count_nonzero(contributions[:, 1]) == 0
    torch.testing.assert_close(logits, base + contributions.sum(1))
    for bits in itertools.product((0., 1.), repeat=4):
        retention = torch.tensor([bits])
        kept, _ = fuse_logits(base, corrections, gates, available, retention)
        expected = base + (contributions * retention[:, :, None, None, None, None]).sum(1)
        torch.testing.assert_close(kept, expected)
    result = verify_stability_enumeration(base[0], contributions[0])
    assert result["passed"], result


def check_utility_sign_and_detachment():
    base = torch.zeros(1, 4, 3, 3, 3, requires_grad=True)
    correction = torch.zeros(1, 4, 3, 3, 3, 3)
    correction[:, 0, 2] = 1.  # Helps true class ET; removal must increase CE.
    correction[:, 1, 2] = -0.2  # Harms true ET in the same context.
    correction.requires_grad_()
    available = torch.ones(1, 4)
    labels = torch.full((1, 3, 3, 3), 3, dtype=torch.long)
    target, delta = utility_targets(base, correction, available, labels)
    assert bool((target[:, 0] > .5).all()) and bool((target[:, 1] < .5).all())
    assert bool((target[:, 2:] == .5).all())
    assert not target.requires_grad and not delta.requires_grad
    q = torch.full_like(target, .5, requires_grad=True)
    loss = balanced_gate_bce(q, target, labels, available)
    loss.backward()
    assert torch.isfinite(q.grad).all() and base.grad is None and correction.grad is None


def check_model_and_gradients():
    model = DetailTrustNet(checkpoint_blocks=True)
    assert parameter_counts(model)["total"] == 16934644
    images = torch.randn(1, 4, 24, 24, 24)
    labels = torch.randint(0, 4, (1, 16, 16, 16))
    available = torch.ones(1, 4)
    output = model(images, available)
    assert output["logits"].shape == (1, 4, 16, 16, 16)
    assert output["corrections"].shape == (1, 4, 3, 16, 16, 16)
    assert output["gates"].shape == (1, 4, 1, 16, 16, 16)
    loss = segmentation_loss(output["logits"], labels)
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    for module in (model.backbone, model.splitter, model.refiner, model.gate):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def check_frozen_stage_and_roundtrip():
    model = DetailTrustNet(checkpoint_blocks=False)
    variant = experiment_registry()["B5"]
    configure_stage(model, "C", variant)
    assert all(not p.requires_grad for m in (model.backbone, model.splitter, model.refiner) for p in m.parameters())
    assert all(p.requires_grad for p in model.gate.parameters())
    images, available = torch.randn(1, 4, 24, 24, 24), torch.ones(1, 4)
    output = model(images, available)
    assert not output["corrections"].requires_grad and output["gates"].requires_grad
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)/"state.pt"
        torch.save(model.state_dict(), path)
        rebuilt = DetailTrustNet(checkpoint_blocks=False)
        rebuilt.load_state_dict(torch.load(path, weights_only=True), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(output["logits"], rebuilt(images, available)["logits"])


def check_patch_stream_and_metrics():
    store = SyntheticStore()
    subjects = list(store.records)[:4]
    dataset = PatchDataset(store, subjects, (16, 16, 16), 22, 10)
    batch = dataset[3]
    resumed = PatchDataset(store, subjects, (16, 16, 16), 22, 7, start=3)[0]
    for key in ("images", "labels", "available", "resolution"):
        torch.testing.assert_close(batch[key], resumed[key])
    assert batch["images"].shape == (4, 24, 24, 24)
    assert batch["labels"].shape == (16, 16, 16)
    empty = np.zeros((5, 5, 5), bool)
    assert binary_metrics(empty, empty, (1, 1, 1))["dice"] == 1
    single = empty.copy(); single[2, 2, 2] = True
    assert binary_metrics(single, empty, (1, 1, 1))["hd95_mm"] is None
    shifted = np.roll(single, 1, axis=0)
    assert binary_metrics(single, shifted, (2, 1, 1))["hd95_mm"] == 2
    assert lesion_metrics(single, single)["lesion_recall"] == 1
    for family, severity in (("resolution", 2.), ("noise", .1), ("bias", .1), ("ghost", .2), ("ringing", .2)):
        image, _, _ = store.load(subjects[0])
        damaged, _ = corrupt_image(image[0], family, severity, np.random.default_rng(1))
        assert damaged.shape == image[0].shape and np.isfinite(damaged).all()


def check_sliding_identity_and_leakage():
    store = SyntheticStore(size=16)
    cfg = configuration("smoke", patch=(16, 16, 16), synthetic=True)
    images, labels, _ = store.load("synthetic_000")
    model = DetailTrustNet(checkpoint_blocks=False)
    result = sliding_window(model, images, np.ones(4, np.float32), cfg, torch.device("cpu"), analysis=True)
    assert result["logits"].shape == (4, 16, 16, 16)
    torch.testing.assert_close(result["logits"], result["base"] + result["contributions"].sum(0))
    split = make_split([dict(id=s) for s in store.records], 5)
    try:
        evaluate_condition(model, store, [split["train"][0]], split, "validation", {}, "test", cfg, torch.device("cpu"))
        raise RuntimeError("Training subject entered held-out evaluation")
    except AssertionError:
        pass


def check_json_and_registry():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)/"report.json"
        write_json(path, dict(undefined=float("nan"), infinity=float("inf"), count=np.int64(3)))
        assert json.loads(path.read_text()) == dict(undefined=None, infinity=None, count=3)
    registry = experiment_registry()
    assert set(registry) == {"B0", "B1", "B2", "B3", "B4", "B5", "A1", "A2", "A3", "A4", "A5", "A6", "A7"}
    assert registry["B4"]["utility"] is False and registry["B5"]["utility"] is True
    for key, variant in registry.items():
        from .model import build_model
        model = build_model(variant, checkpoint_blocks=False)
        with torch.no_grad():
            output = model(torch.randn(1, 4, 24, 24, 24), torch.tensor([[1., 0., 1., 1.]]))
        assert output["logits"].shape == (1, 4, 16, 16, 16)
        assert torch.isfinite(output["logits"]).all()
        assert torch.count_nonzero(output["contributions"][:, 1]) == 0


def check_nifti_adapter():
    """Actual NIfTI read/reorientation test when nibabel is installed."""
    if importlib.util.find_spec("nibabel") is None:
        return dict(status="blocked", reason="nibabel not installed in local CPU environment")
    import nibabel as nib
    from .data import discover_subjects, load_subject
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        subject = "BraTS20_Training_001"
        affine = np.diag([-1., 1., 1., 1.]); affine[0, 3] = 15
        for name in ("t1", "t1ce", "t2", "flair", "seg"):
            array = np.zeros((16, 16, 16), np.float32)
            array[2:9, 2:9, 2:9] = 4 if name == "seg" else 2
            nib.save(nib.Nifti1Image(array, affine), root/f"{subject}_{name}.nii.gz")
        records, _ = discover_subjects(root)
        images, labels, meta = load_subject(records[0])
        assert images.shape == (4, 16, 16, 16) and labels.max() == 3 and meta["orientation"] == "RAS"
    return dict(status="passed")


def run_unit_checks():
    checks = [check_labels_and_splits, check_gaussian_and_haar, check_availability_and_anchor,
              check_fusion_and_stability, check_utility_sign_and_detachment, check_model_and_gradients,
              check_frozen_stage_and_roundtrip, check_patch_stream_and_metrics,
              check_sliding_identity_and_leakage, check_json_and_registry, check_nifti_adapter]
    results = []
    seed_everything(7)
    for check in checks:
        started = time.perf_counter()
        value = check()
        result = dict(name=check.__name__, seconds=time.perf_counter()-started, **(value or {"status": "passed"}))
        results.append(result)
        print(f"{check.__name__}: {result['status']}", flush=True)
    return results


def tiny_overfit_check(store, subjects, cfg, device, updates=20):
    """Repeated one-patch B0 optimization; diagnostic only, never an evaluation score."""
    from .model import build_model
    variant = experiment_registry()["B0"]
    model = build_model(variant, cfg.checkpoint_blocks).to(device)
    configure_stage(model, "RAW", variant)
    optimizer, scheduler = make_optimizer(model, "RAW", updates, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg.precision == "amp_fp16")
    batch = next(iter(make_loader(PatchDataset(store, subjects[:2], cfg.patch, cfg.seed, 1, 0.))))
    losses = []
    for _ in range(updates):
        losses.append(train_update(model, batch, optimizer, scaler, "RAW", cfg, variant, device)["total"])
        scheduler.step()
    return dict(status="completed", losses=losses, loss_decreased=losses[-1] < losses[0],
                interpretation="Repeated training patch sanity check, not held-out performance")
