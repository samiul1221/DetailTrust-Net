"""Strict JSON reports, honest artifact manifests, and plots from actual records."""
from dataclasses import asdict
from pathlib import Path
import csv
import json
import platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import write_json, safe_json, environment_record, experiment_registry, LABEL_MAPPING, MODALITIES
from .metrics import aggregate_metrics, paired_patient_comparison
from .paper_figures import generate_paper_figures

REPORT_FIELDS = ("schema_version", "project", "architecture", "architecture_equations", "implementation_deviations",
                 "environment", "hardware", "dataset", "dataset_integrity", "label_mapping", "preprocessing",
                 "data_splits", "random_seeds", "training_configuration", "training_stages", "training_history",
                 "best_checkpoint", "model_parameters", "model_complexity", "validation_metrics", "test_metrics",
                 "per_subject_metrics", "class_metrics", "region_metrics", "missing_modality_robustness",
                 "spatial_corruption_robustness", "detail_branch_contributions", "utility_gate_analysis",
                 "stability_analysis", "ablation_study", "statistical_analysis", "runtime_and_memory",
                 "reproducibility", "warnings", "limitations", "experiment_status")


def implementation_decisions():
    return [
        "B3 means four trainable global modality sigmoid scalars, learned in Stage C; no image-dependent pooling network.",
        "A1 permutes targets spatially within each subject/modality; target marginal distribution is retained.",
        "A4 is one-level Haar low-band inverse reconstruction using pairwise averaging/repetition; even origin, no learned filters.",
        "A6 unfreezes all components in Stage C after the same A/B initialization; detached targets still recompute each step.",
        "B2 receives an extra B-style refinement stage of C updates, matching total update count; gate controls fork the PRE-refinement B checkpoint.",
        "Utility average pooling excludes padded positions at boundaries, an explicit resolution of unspecified pooling padding.",
        "A7 uses the declared central-patch coarse lattice and adaptive-average occupancy operator for globally resolution-degraded samples only. It does not assert equivalence to the entire blur/resample MRI simulator or local blend mask.",
        "The BraTS20 adapter reorients losslessly to RAS and rejects non-1mm or mismatched affines. It does not silently resample another dataset.",
        "Patient lists from the cited author protocol are not recovered; the notebook uses a saved custom subject split.",
        "Surface Dice uses unweighted 6-connected boundary voxel centres, not area-weighted surfels; HD95 concatenates directed distances.",
        "Stage transitions restore the best validation checkpoint; epochs are fixed update-reporting intervals, not complete dataset passes.",
        "Main/extended start at 96 cubed; CPU synthetic tests explicitly use 16 cubed while preserving every model width.",
        "Training augmentations use aligned flips without geometric interpolation; no additional rotations are required by the starting protocol.",
        "Gate and candidate maps are window averages. Their product is not used as the final contribution; c is independently blended.",
    ]


def write_csv(path, rows):
    """No fake header-only metric files: return a reason if there are no records."""
    if not rows:
        return dict(status="not_produced", reason="No measured records")
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            cleaned = safe_json(row)
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in cleaned.items()})
    return dict(status="saved", path=str(path), rows=len(rows))


def collect_results(statuses):
    records = dict(rows=[], gates=[], corrections=[], stability=[], stability_checks=[], timings=[], history=[])
    for experiment, status in statuses.items():
        if not status.get("stages"):
            for source, history in status.get("partial_history", {}).items():
                records["history"] += [dict(row, experiment=experiment, source=source, incomplete=True) for row in history]
        for stage in status.get("stages", []):
            records["history"] += [dict(row, experiment=experiment, shared=stage.get("shared", False)) for row in stage["history"]]
        for split in ("validation", "test"):
            for condition in status.get(split, {}).get("results", []):
                for key in ("rows", "gates", "corrections", "stability", "stability_checks", "timings"):
                    records[key] += condition.get(key, [])
    return records


def generate_plots(records, summary, output_dir):
    """Render measured patient results in PNG/PDF/SVG, with caption/data sidecars."""
    return generate_paper_figures(records, output_dir)


def build_report(cfg, split, integrity, statuses, best_checkpoint, checks, profiles, dataset_evidence):
    records = collect_results(statuses)
    summary = aggregate_metrics(records["rows"])
    environment = environment_record(cfg.output_dir)
    comparisons = {f"{seed}_{partition}_{endpoint}": paired_patient_comparison(records["rows"], f"B5_s{seed}", f"B4_s{seed}",
                   partition=partition, endpoint=endpoint)
                   for seed in cfg.seeds for partition in ("validation", "test") for endpoint in ("overall", "degraded", "clean_ET")}
    for key, status in statuses.items():
        comparator = f"B4_s{status['seed']}" if status["variant_id"] in ("B5", "A1") else f"B1_s{status['seed']}"
        status["comparison"] = paired_patient_comparison(records["rows"], key, comparator)
        status["seed_count_for_this_record"] = 1 if status.get("checkpoint") else 0
    report = {key: None for key in REPORT_FIELDS}
    report.update(schema_version="1.0.0", project=dict(name="DetailTrust-Net", evidence_level=cfg.mode,
                  synthetic=cfg.synthetic, hypothesis="Utility-supervised removable detail corrections improve segmentation versus segmentation-only gates"),
        architecture=dict(widths=[32, 64, 128, 256, 320], modalities=list(MODALITIES), classes=4,
                          reference_parameters=16934644, correction_input_channels=37, gate_input_channels=44),
        architecture_equations=dict(split="H=X-sum softmax(alpha)*G_sigma(X)", anchor="r=R(F,H,e)-R(F,0,e)",
                 fusion="z_foreground=z0_foreground+sum a*q*r; z_background=z0_background",
                 utility="sigmoid(AvgPool3(CE(z_all-r,Y)-CE(z_all,Y))/0.1), detached",
                 stability="B_j=(z_winner-z_j)-sum_m max(0,c_m,winner-c_m,j)"),
        implementation_deviations=implementation_decisions(), environment=environment,
        hardware={k: v for k, v in environment.items() if k.startswith("gpu") or k == "cuda"},
        dataset=dataset_evidence, dataset_integrity=integrity, label_mapping=LABEL_MAPPING,
        preprocessing=dict(orientation="RAS", spacing_mm=[1, 1, 1], interpolation="lossless reorientation; labels never interpolated; image degradation uses trilinear",
                           normalization="per-modality nonzero foreground z-score", patch=cfg.patch, halo=4, cache_gb=cfg.cache_gb,
                           crop="available MRI foreground, eight-voxel margin; never labels"),
        data_splits=split, random_seeds=dict(training=cfg.seeds, split=cfg.split_seed), training_configuration=asdict(cfg),
        training_stages={k: [{a: b for a, b in s.items() if a != "history"} for s in v.get("stages", [])] for k, v in statuses.items()},
        training_history=records["history"], best_checkpoint=best_checkpoint,
        model_parameters={k: v.get("parameters") for k, v in statuses.items()},
        model_complexity=dict(macs=None, flops=None, reason="No validated profiler result; no theoretical estimate presented as measured"),
        validation_metrics=[r for r in summary if r["split"] == "validation"],
        test_metrics=dict(status="completed_or_partial" if cfg.run_final_test else "not_run", records=[r for r in summary if r["split"] == "test"]),
        per_subject_metrics=records["rows"], class_metrics=[r for r in summary if r["region"] not in ("WT", "TC", "ET")],
        region_metrics=[r for r in summary if r["region"] in ("WT", "TC", "ET")],
        missing_modality_robustness=[r for r in summary if r["condition"].startswith("clean_m") and "_remove" not in r["condition"] and "base_only" not in r["condition"]],
        spatial_corruption_robustness=[r for r in summary if not r["condition"].startswith("clean_m")],
        detail_branch_contributions=records["corrections"], utility_gate_analysis=records["gates"],
        stability_analysis=dict(records=records["stability"], enumeration_checks=records["stability_checks"],
                                scope="fixed computed contributions; not correctness or whole-input removal"),
        ablation_study=statuses, statistical_analysis=dict(paired_patient_comparisons=comparisons, requested_seed_count=len(cfg.seeds),
                 completed_seed_counts={v: sum(r.get("checkpoint") is not None for r in statuses.values() if r["variant_id"] == v) for v in experiment_registry()},
                 limitation="Single seed is preliminary; bootstrap conditions on weights. No voxel-level independent-sample inference."),
        runtime_and_memory=dict(profiles=profiles, full_volume_timings=records["timings"],
                               inference_benchmarks={k: v.get("inference_benchmark") for k, v in statuses.items()}),
        reproducibility=dict(checks=checks, json_nonfinite_policy="null, with undefined-case counts", cuda_determinism="warn_only; exact bitwise cross-device reproducibility not promised",
            resume_restore=json.loads((Path(cfg.output_dir)/'resume_restore.json').read_text()) if (Path(cfg.output_dir)/'resume_restore.json').exists() else None),
        warnings=["No improvement or novelty claim follows from a smoke/pilot run.", "A large gate or correction is not necessarily useful.",
                  "Mounted dataset licence/version must be checked against dataset card and original BraTS terms."],
        limitations=["Synthetic corruption is not clinical missingness or scanner-domain validation.",
                     "No external patient cohort is evaluated.", "Published M2FTrans/evidential/SwitchNet code is not reproduced in this notebook; internal controls alone do not support SOTA claims.",
                     "Labels are one released consensus per subject, not independent multi-rater annotations.",
                     "Patient subset/one seed results do not establish convergence or generalization."],
        experiment_status={k: dict(status=v["status"], reason=v.get("reason")) for k, v in statuses.items()})
    return report, records, summary


def export_report(report, records, summary, output_dir):
    directory = Path(output_dir)
    write_json(directory/"detailtrust_report.json", report)
    write_json(directory/"training_history.json", records["history"])
    write_json(directory/"ablation_results.json", report["ablation_study"])
    write_json(directory/"stability_analysis.json", report["stability_analysis"])
    artifacts = {}
    for name, rows in (("metrics_per_subject.csv", records["rows"]), ("metrics_summary.csv", summary),
                       ("gate_analysis.csv", records["gates"]), ("correction_contributions.csv", records["corrections"])):
        artifacts[name] = write_csv(directory/name, rows)
    artifacts["figures"] = generate_plots(records, summary, directory)
    parsed = json.loads((directory/"detailtrust_report.json").read_text())
    assert set(REPORT_FIELDS) <= set(parsed)
    write_json(directory/"artifact_manifest.json", artifacts)
    return artifacts


def verify_artifacts(output_dir):
    names = ["best_detailtrust_model.pt", "best_detailtrust_model_config.json", "detailtrust_report.json",
             "training_history.json", "ablation_results.json", "split_manifest.json", "experiment_config.json",
             "metrics_per_subject.csv", "metrics_summary.csv", "gate_analysis.csv", "correction_contributions.csv",
             "stability_analysis.json", "README.md"]
    result = {name: dict(exists=(Path(output_dir)/name).is_file(),
                        bytes=(Path(output_dir)/name).stat().st_size if (Path(output_dir)/name).is_file() else None)
              for name in names}
    write_json(Path(output_dir)/"artifact_verification.json", result)
    return result
