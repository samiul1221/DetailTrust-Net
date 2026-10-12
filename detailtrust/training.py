"""Explicit stages, AMP training, validation selection, and resumable checkpoints."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import random
import time
import numpy as np
import torch
from .config import write_json, safe_json, file_hash, seed_everything
from .data import PatchDataset, make_loader
from .losses import stage_objective
from .evaluation import autocast_context, synchronized_time, validation_score
from .model import build_model, parameter_counts


class SessionBudgetReached(RuntimeError):
    """The latest checkpoint is saved; continue with the same output directory."""


def configure_stage(model, stage, variant):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.freeze_candidates = stage == "C" and not variant["joint"]
    if stage in ("A", "B", "RAW") or variant["joint"]:
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(True)
        if model.splitter:
            for parameter in model.splitter.parameters():
                parameter.requires_grad_(True)
    if stage == "B" or (stage == "C" and variant["joint"]):
        for parameter in model.refiner.parameters():
            parameter.requires_grad_(True)
    if stage == "C":
        if model.gate:
            for parameter in model.gate.parameters():
                parameter.requires_grad_(True)
        if model.global_logits is not None:
            model.global_logits.requires_grad_(True)
    assert any(p.requires_grad for p in model.parameters()), "This stage has no trainable parameters."


def make_optimizer(model, stage, updates, cfg):
    lr = cfg.gate_lr if stage == "C" else cfg.lr
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=cfg.weight_decay)
    warmup = min(cfg.warmup_updates, max(1, updates // 10))
    def multiplier(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1., (step - warmup) / max(1, updates - warmup))
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    return optimizer, scheduler


def random_states():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_random_states(states):
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"].cpu())
    if torch.cuda.is_available() and states["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in states["cuda"]])


def save_checkpoint(path, model, optimizer, scheduler, scaler, stage, update, score, best_score,
                    cfg, variant, split, history, fingerprint, stale=0, elapsed=0.):
    """Full checkpoints are trusted local artifacts; never load arbitrary pickles."""
    path = Path(path)
    payload = dict(format_version=1, model_state=model.state_dict(), architecture=model.architecture_config,
                   optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
                   scaler_state=scaler.state_dict(), stage=stage, update=update,
                   epoch=(update + cfg.epoch_updates - 1) // cfg.epoch_updates,
                   validation_score=score, best_validation_score=best_score,
                   selection_criterion="mean patient Dice across WT,TC,ET on clean validation",
                   seed=cfg.seed, config=asdict(cfg), variant=variant, split_manifest=split,
                   history=history, rng=random_states(), fingerprint=fingerprint,
                   stale_validations=stale, elapsed_training_seconds=elapsed)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path, model, device, optimizer=None, scheduler=None, scaler=None, fingerprint=None):
    checkpoint_data = torch.load(path, map_location=device, weights_only=False)
    if fingerprint is not None and checkpoint_data["fingerprint"] != fingerprint:
        raise ValueError("Resume configuration/split mismatch; use a new output directory.")
    model.load_state_dict(checkpoint_data["model_state"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_data["optimizer_state"])
        scheduler.load_state_dict(checkpoint_data["scheduler_state"])
        scaler.load_state_dict(checkpoint_data["scaler_state"])
        restore_random_states(checkpoint_data["rng"])
    return checkpoint_data


def train_update(model, batch, optimizer, scaler, stage, cfg, variant, device):
    """One optimizer update; no hidden gradient accumulation or clipping."""
    model.train()
    images = batch["images"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    available = batch["available"].to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, cfg.precision):
        outputs = model(images, available, stage="A" if stage == "RAW" else stage)
    # Explicitly outside autocast: CE, softmax, pooling and reductions are float32.
    loss, terms = stage_objective(outputs, labels, stage, cfg, variant, batch["resolution"])
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite training loss")
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return terms


def training_fingerprint(cfg, variant, split, stage, updates, subjects):
    keys = ("patch", "batch_size", "precision", "seed", "lr", "gate_lr", "weight_decay",
            "branch_dropout", "utility_tau", "utility_weight", "corruption_probability",
            "warmup_updates", "validation_every", "val_subjects", "synthetic")
    payload = dict(config={k: getattr(cfg, k) for k in keys}, variant=variant,
                   split=split["sha256"], stage=stage, updates=updates, subjects=subjects)
    return hashlib.sha256(json.dumps(safe_json(payload), sort_keys=True).encode()).hexdigest()


def run_training_stage(model, store, subjects, validation_subjects, split, cfg, variant,
                       stage, updates, directory, device, deadline, stream_offset=0):
    """Resume latest; select best validation weights separately from last update."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    latest, best = directory / "latest.pt", directory / "best.pt"
    configure_stage(model, stage, variant)
    optimizer, scheduler = make_optimizer(model, stage, updates, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg.precision == "amp_fp16")
    fingerprint = training_fingerprint(cfg, variant, split, stage, updates, subjects)
    completed, history, best_score, stale, prior_seconds = 0, [], -float("inf"), 0, 0.
    if latest.exists():
        state = load_checkpoint(latest, model, device, optimizer, scheduler, scaler, fingerprint)
        completed, history = state["update"], state["history"]
        best_score, stale, prior_seconds = state["best_validation_score"], state["stale_validations"], state["elapsed_training_seconds"]
    dataset = PatchDataset(store, subjects, cfg.patch, cfg.seed, max(0, updates-completed),
                           cfg.corruption_probability if variant["corruption"] else 0., start=stream_offset + completed)
    loader = make_loader(dataset)
    started, saved = time.perf_counter(), time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    score = None
    for batch in loader:
        if time.perf_counter() >= deadline:
            save_checkpoint(latest, model, optimizer, scheduler, scaler, stage, completed, score, best_score,
                            cfg, variant, split, history, fingerprint, stale, prior_seconds + time.perf_counter()-started)
            write_json(directory / "history.json", history)
            raise SessionBudgetReached(f"Paused {stage} at {completed}/{updates}; resume {latest}")
        before = synchronized_time(device)
        terms = train_update(model, batch, optimizer, scaler, stage, cfg, variant, device)
        scheduler.step()
        completed += 1
        seconds = synchronized_time(device) - before
        row = dict(stage=stage, update=completed, epoch=(completed-1)//cfg.epoch_updates+1,
                   subject=batch["subject"][0], patch_seconds=seconds,
                   patches_per_second=1/seconds, learning_rate=optimizer.param_groups[0]["lr"], **terms)
        validate_now = completed % cfg.validation_every == 0 or completed == updates
        if validate_now:
            val_start = synchronized_time(device)
            score, val_loss = validation_score(model, store, validation_subjects, split, cfg, device,
                                                "A" if stage == "RAW" else stage)
            row.update(validation_dice=score, validation_loss=val_loss,
                       validation_seconds=synchronized_time(device)-val_start)
            improved = score > best_score
            best_score, stale = max(score, best_score), 0 if improved else stale+1
        history.append(row)
        elapsed = prior_seconds + time.perf_counter() - started
        if validate_now or time.perf_counter()-saved >= cfg.checkpoint_seconds:
            save_checkpoint(latest, model, optimizer, scheduler, scaler, stage, completed, score, best_score,
                            cfg, variant, split, history, fingerprint, stale, elapsed)
            write_json(directory / "history.json", history)
            saved = time.perf_counter()
            if validate_now and improved:
                save_checkpoint(best, model, optimizer, scheduler, scaler, stage, completed, score, best_score,
                                cfg, variant, split, history, fingerprint, stale, elapsed)
        if completed % cfg.epoch_updates == 0 or validate_now:
            print(f"{directory.parent.name} {stage} {completed}/{updates}: loss={terms['total']:.4f}, best_val={best_score:.4f}", flush=True)
        if cfg.early_stopping_patience and stale >= cfg.early_stopping_patience:
            break
    if not best.exists():
        raise RuntimeError("No validation-selected checkpoint exists; complete at least one validation.")
    state = load_checkpoint(best, model, device)
    result = dict(status="completed" if completed == updates else "early_stopped", stage=stage,
                  updates_completed=completed, updates_planned=updates, history=history,
                  checkpoint=str(best), checkpoint_sha256=file_hash(best), checkpoint_bytes=best.stat().st_size,
                  selected_update=state["update"], selected_epoch=state["epoch"], validation_score=state["validation_score"],
                  training_seconds=prior_seconds + time.perf_counter()-started,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                  parameter_counts=parameter_counts(model))
    write_json(directory / "stage_result.json", result)
    return result


def train_stage_a(*args, **kwargs):
    return run_training_stage(*args, stage="A", **kwargs)


def train_stage_b(*args, **kwargs):
    return run_training_stage(*args, stage="B", **kwargs)


def train_stage_c(*args, **kwargs):
    return run_training_stage(*args, stage="C", **kwargs)


def profile_training_stages(store, subjects, cfg, variant, device):
    """Disposable full models; measured optimizer steps include fetching each patch."""
    rows = []
    for stage in ("A", "B", "C"):
        seed_everything(cfg.seed)
        model = build_model(variant, cfg.checkpoint_blocks).to(device)
        configure_stage(model, stage, variant)
        total = cfg.profile_warmup + cfg.profile_updates
        optimizer, _ = make_optimizer(model, stage, total, cfg)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg.precision == "amp_fp16")
        dataset = PatchDataset(store, subjects, cfg.patch, cfg.seed, total)
        iterator, times = iter(make_loader(dataset)), []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for index in range(total):
            before = synchronized_time(device)
            batch = next(iterator)
            train_update(model, batch, optimizer, scaler, stage, cfg, variant, device)
            elapsed = synchronized_time(device)-before
            if index >= cfg.profile_warmup:
                times.append(elapsed)
        rows.append(dict(stage=stage, warmup=cfg.profile_warmup, measured_updates=len(times),
                         median_step_seconds=float(np.median(times)), p90_step_seconds=float(np.quantile(times, .9)),
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                         peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                         device=str(device), patch=list(cfg.patch), includes_loading=True))
        del optimizer, model, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows
