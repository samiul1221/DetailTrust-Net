"""Central experiment execution, shared candidate checkpoints, and evaluation status."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import shutil
import time
import numpy as np
import torch
from .config import experiment_registry, seed_everything, write_json, file_hash
from .model import build_model, parameter_counts, DetailTrustNet
from .training import run_training_stage, SessionBudgetReached, load_checkpoint
from .evaluation import evaluation_conditions, evaluate_condition, sliding_window, synchronized_time, benchmark_inference


def copy_candidate_weights(target, source_checkpoint, device):
    """Strict per-module copies preserve shared Stage B candidates across gate controls."""
    source = torch.load(source_checkpoint, map_location=device, weights_only=False)
    state = source["model_state"]
    for name in ("backbone", "splitter", "refiner", "gate"):
        module = getattr(target, name)
        if module is not None:
            prefix = name + "."
            child = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            module.load_state_dict(child, strict=True)


def train_candidate(model, store, train_ids, val_ids, split, cfg, variant, folder, device, deadline):
    """Run A then B; restoring each best checkpoint is part of the declared protocol."""
    stages = []
    for stage, updates, offset in (("A", cfg.updates_a, 0), ("B", cfg.updates_b, cfg.updates_a)):
        stages.append(run_training_stage(model, store, train_ids, val_ids, split, cfg, variant,
                                         stage, updates, Path(folder)/stage, device, deadline, offset))
    return stages


def evaluate_experiment(model, store, subjects, split, partition, experiment, cfg, device, folder, checkpoint_hash, deadline):
    """One JSON per condition supports interrupted robustness sweeps."""
    condition_results, condition_status = [], []
    for condition in evaluation_conditions(cfg, final=partition == "test"):
        path = Path(folder)/partition/(condition["id"] + ".json")
        expected = dict(subjects=subjects, condition=condition, checkpoint_hash=checkpoint_hash,
                        split_hash=split["sha256"], patch=list(cfg.patch), surface_tolerance=cfg.surface_tolerance_mm,
                        small_et_threshold_mm3=cfg.small_et_threshold_mm3)
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["identity"] != expected:
                raise ValueError("Existing evaluation belongs to a different configuration/checkpoint.")
            result = saved["result"]
        elif time.perf_counter() < deadline:
            result = evaluate_condition(model, store, subjects, split, partition, condition, experiment,
                                        cfg, device, analysis=condition["id"] == "clean_m15")
            write_json(path, dict(identity=expected, result=result))
        else:
            condition_status.append(dict(condition=condition, status="not_run", reason="session time cap; resume evaluation"))
            continue
        condition_results.append(result)
        condition_status.append(dict(condition=condition, status="completed"))
    return dict(status="completed" if all(c["status"] == "completed" for c in condition_status) else "partial",
                conditions=condition_status, results=condition_results)


def run_experiment_sweep(store, split, cfg, device):
    """Core order B0/B1/B2/B4/B5; all secondary variants use the same pipeline."""
    registry = experiment_registry()
    root = Path(cfg.output_dir)/"experiments"
    root.mkdir(parents=True, exist_ok=True)
    statuses = {f"{variant}_s{seed}": dict(variant_id=variant, seed=seed, description=definition["description"],
                 configuration=definition, status="not_run", reason="not selected in this allocation")
                for seed in cfg.seeds for variant, definition in registry.items()}
    started = time.perf_counter()
    deadline = started + cfg.max_hours * 3600
    training_deadline = started + cfg.max_hours * 3600 * .7
    train_ids = split["train"][:min(cfg.train_subjects, len(split["train"]))]
    val_ids = split["validation"][:cfg.val_subjects]
    for seed in cfg.seeds:
        seeded = replace(cfg, seed=seed)
        shared_checkpoint, shared_stages = None, []
        for variant_id in cfg.variants:
            experiment = f"{variant_id}_s{seed}"
            variant = registry[variant_id]
            folder = root/experiment
            folder.mkdir(exist_ok=True)
            record = statuses[experiment]
            record.update(reason=None, exact_run_configuration=asdict(seeded), train_subjects=train_ids,
                          validation_subjects=val_ids, split_hash=split["sha256"], evidence_level=cfg.mode,
                          scientific_comparison_status="preliminary; assess convergence and repeated seeds")
            status_path = folder/"result.json"
            seed_everything(seed)
            model = None
            try:
                if time.perf_counter() >= deadline:
                    record.update(status="not_run", reason="session allocation exhausted")
                    continue
                model = build_model(variant, cfg.checkpoint_blocks).to(device)
                record["parameters"] = parameter_counts(model)
                stages = []
                if variant["gate"] == "raw":
                    stages.append(run_training_stage(model, store, train_ids, val_ids, split, seeded, variant,
                                  "RAW", cfg.updates_a+cfg.updates_b+cfg.updates_c, folder/"RAW", device, training_deadline))
                else:
                    if variant_id in ("B2", "B3", "B4", "B5", "A1", "A6"):
                        if shared_checkpoint is None:
                            seed_everything(seed)  # Independent of which gate variant runs first.
                            candidate = build_model(registry["B5"], cfg.checkpoint_blocks).to(device)
                            try:
                                shared_stages = train_candidate(candidate, store, train_ids, val_ids, split, seeded,
                                                registry["B5"], root/f"shared_candidates_s{seed}", device, training_deadline)
                            finally:
                                del candidate
                            shared_checkpoint = shared_stages[-1]["checkpoint"]
                        copy_candidate_weights(model, shared_checkpoint, device)
                        stages = [dict(s, shared=True) for s in shared_stages]
                        record["shared_stage_b_checkpoint"] = shared_checkpoint
                        # Same gate initialization and patch stream for B4/B5/A1.
                        seed_everything(seed + 20000)
                    else:
                        stages = train_candidate(model, store, train_ids, val_ids, split, seeded, variant,
                                                  folder, device, training_deadline)
                    # Give ungated B2 the same total number of updates, without changing gates.
                    stage = "B" if variant_id == "B2" else "C"
                    stages.append(run_training_stage(model, store, train_ids, val_ids, split, seeded, variant,
                                   stage, cfg.updates_c, folder/"refinement", device, training_deadline,
                                   cfg.updates_a+cfg.updates_b))
                selected = stages[-1]
                record.update(status="trained", stages=stages, checkpoint=selected["checkpoint"],
                              checkpoint_sha256=selected["checkpoint_sha256"],
                              validation_selection_score=selected["validation_score"],
                              training_seconds=sum(s["training_seconds"] for s in stages),
                              incremental_training_seconds=sum(s["training_seconds"] for s in stages if not s.get("shared")))
                write_json(status_path, record)
                benchmark_path = folder/"inference_benchmark.json"
                benchmark_identity = dict(checkpoint_sha256=selected["checkpoint_sha256"],
                    subject=val_ids[0], patch=list(cfg.patch), precision=cfg.precision,
                    repeats=cfg.inference_repeats, device=str(device))
                if benchmark_path.exists():
                    benchmark_saved = json.loads(benchmark_path.read_text())
                    if benchmark_saved.get("identity") != benchmark_identity:
                        raise ValueError("Inference benchmark configuration changed; use a fresh output directory.")
                    record["inference_benchmark"] = benchmark_saved["result"]
                elif time.perf_counter() < deadline:
                    record["inference_benchmark"] = benchmark_inference(model, store, val_ids[0], cfg, device)
                    write_json(benchmark_path, dict(identity=benchmark_identity, result=record["inference_benchmark"]))
                else:
                    record["inference_benchmark"] = dict(status="not_run", reason="time cap")
                record["validation"] = evaluate_experiment(model, store, val_ids, split, "validation", experiment,
                                                           cfg, device, folder, selected["checkpoint_sha256"], deadline)
                if cfg.run_final_test:
                    record["test"] = evaluate_experiment(model, store, split["test"], split, "test", experiment,
                                                         cfg, device, folder, selected["checkpoint_sha256"], deadline)
                else:
                    record["test"] = dict(status="not_run", reason="test remains sealed; enable only with locked main protocol")
                record["status"] = "completed" if record["validation"]["status"] == "completed" and record["test"]["status"] in ("completed", "not_run") else "evaluation_partial"
            except SessionBudgetReached as error:
                record.update(status="paused", reason=str(error))
            except torch.cuda.OutOfMemoryError:
                record.update(status="failed", reason="GPU OOM; no automatic architecture change. Profile checkpointing or explicitly use matched 80³ patches.")
            except Exception as error:
                record.update(status="failed", reason=f"{type(error).__name__}: {error}")
            finally:
                if record["status"] in ("paused", "failed"):
                    record["partial_history"] = {str(p.relative_to(root)): json.loads(p.read_text())
                        for directory in (folder, root/f"shared_candidates_s{seed}") if directory.exists()
                        for p in directory.rglob("history.json")}
                if model is not None:
                    del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                write_json(status_path, record)
                write_json(Path(cfg.output_dir)/"ablation_results.json", statuses)
    return statuses


def export_best_detailtrust(statuses, output_dir, device):
    """Choose B5 seed by validation only. No untrained checkpoint placeholders."""
    candidates = [r for r in statuses.values() if r["variant_id"] == "B5" and r.get("checkpoint")]
    if not candidates:
        return dict(status="not_produced", reason="No B5 training stage completed")
    selected = max(candidates, key=lambda r: r["validation_selection_score"])
    source = selected["checkpoint"]
    destination = Path(output_dir)/"best_detailtrust_model.pt"
    shutil.copy2(source, destination)
    state = torch.load(destination, map_location="cpu", weights_only=False)
    rebuilt = DetailTrustNet(**state["architecture"])
    rebuilt.load_state_dict(state["model_state"], strict=True)
    config = dict(architecture=state["architecture"], variant_id="B5", seed=selected["seed"],
                  label_mapping={"0": 0, "1": 1, "2": 2, "4": 3}, modality_order=list(("t1", "t1ce", "t2", "flair")),
                  preprocessing="RAS; verified 1mm; nonzero z-score; available-image crop; 4-voxel halo")
    write_json(Path(output_dir)/"best_detailtrust_model_config.json", config)
    return dict(status="saved", path=str(destination), sha256=file_hash(destination), size_bytes=destination.stat().st_size,
                selected_epoch=state["epoch"], selected_update=state["update"], seed=selected["seed"],
                validation_score=state["validation_score"], criterion=state["selection_criterion"],
                architecture=state["architecture"], format="PyTorch full resume checkpoint v1",
                strict_state_dict_load="passed", source_experiment=f"B5_s{selected['seed']}")
